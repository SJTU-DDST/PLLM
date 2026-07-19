#include <infiniband/verbs.h>

#ifdef PLLM_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <arpa/inet.h>
#include <endian.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x504c504f;
constexpr std::uint16_t kVersion = 2;
constexpr std::uint64_t kSlotMagic = 0x504c4c4d504f4f4cULL;
constexpr std::size_t kHeaderBytes = 64;
constexpr std::size_t kTransferChunk = 8ULL * 1024 * 1024;
constexpr std::uint64_t kDefaultPoolBytes = 16ULL * 1024 * 1024 * 1024;
constexpr std::uint64_t kDefaultSlotBytes = 3ULL * 1024 * 1024;

enum class Operation : std::uint16_t { Put = 1, Get = 2, StreamGet = 3 };

struct Options {
  bool server = false;
  std::string client;
  std::string index_file;
  std::string root;
  std::string device;
  std::string allocator = "aligned";
  std::string token;
  std::string token_file;
  bool insecure_no_auth = false;
  Operation operation = Operation::Put;
  std::uint16_t port = 17902;
  std::uint8_t ib_port = 1;
  int gid_index = 0;
  std::uint64_t pool_bytes = kDefaultPoolBytes;
  std::uint64_t slot_bytes = kDefaultSlotBytes;
  std::size_t queue_depth = 8;
  std::size_t rd_atomic_depth = 16;
  int stream_shm_fd = -1;
  std::size_t stream_shm_bytes = 0;
};

#pragma pack(push, 1)
struct AuthWire {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t operation;
  std::uint32_t token_size;
};

struct PoolInfoWire {
  std::uint32_t status;
  std::uint32_t message_size;
  std::uint64_t pool_bytes;
  std::uint64_t slot_bytes;
  std::uint64_t slot_count;
};

struct ConnectionWire {
  std::uint16_t lid;
  std::uint8_t mtu;
  std::uint8_t gid_index;
  std::uint8_t rd_atomic_depth;
  std::array<std::uint8_t, 3> reserved;
  std::uint32_t qpn;
  std::uint32_t psn;
  std::uint32_t rkey;
  std::uint64_t address;
  std::array<std::uint8_t, 16> gid;
};

struct SlotHeaderWire {
  std::uint64_t magic;
  std::uint64_t size;
  std::uint64_t key_hash;
  std::uint64_t slot;
  std::uint64_t committed;
  std::array<std::uint8_t, kHeaderBytes - 40> reserved;
};

struct StreamResponseWire {
  std::uint32_t status;
  std::uint64_t payload_size;
  std::uint32_t message_size;
};
#pragma pack(pop)

static_assert(sizeof(SlotHeaderWire) == kHeaderBytes);
static_assert(sizeof(StreamResponseWire) == 16);

struct IndexEntry {
  std::uint64_t slot = 0;
  std::filesystem::path key;
  std::uint64_t size = 0;
};

[[noreturn]] void system_error(const std::string& operation) {
  throw std::runtime_error(operation + ": " + std::strerror(errno));
}

std::string json_escape(std::string_view value) {
  std::string output;
  for (char ch : value) {
    if (ch == '\\' || ch == '"') output.push_back('\\');
    if (ch == '\n') output += "\\n";
    else output.push_back(ch);
  }
  return output;
}

void send_all(int socket_fd, const void* data, std::size_t size) {
  const auto* cursor = static_cast<const std::byte*>(data);
  while (size > 0) {
    const ssize_t sent = ::send(socket_fd, cursor, size, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR) continue;
      system_error("send");
    }
    if (sent == 0) throw std::runtime_error("peer closed while sending");
    cursor += sent;
    size -= static_cast<std::size_t>(sent);
  }
}

void recv_all(int socket_fd, void* data, std::size_t size) {
  auto* cursor = static_cast<std::byte*>(data);
  while (size > 0) {
    const ssize_t received = ::recv(socket_fd, cursor, size, 0);
    if (received < 0) {
      if (errno == EINTR) continue;
      system_error("recv");
    }
    if (received == 0) throw std::runtime_error("peer closed while receiving");
    cursor += received;
    size -= static_cast<std::size_t>(received);
  }
}

bool read_fd_all(int fd, void* data, std::size_t size, bool allow_eof = false) {
  auto* cursor = static_cast<std::byte*>(data);
  std::size_t received = 0;
  while (received < size) {
    const ssize_t count = ::read(fd, cursor + received, size - received);
    if (count < 0) {
      if (errno == EINTR) continue;
      system_error("read stream protocol");
    }
    if (count == 0) {
      if (allow_eof && received == 0) return false;
      throw std::runtime_error("short stream protocol read");
    }
    received += static_cast<std::size_t>(count);
  }
  return true;
}

void write_fd_all(int fd, const void* data, std::size_t size) {
  const auto* cursor = static_cast<const std::byte*>(data);
  while (size > 0) {
    const ssize_t count = ::write(fd, cursor, size);
    if (count < 0) {
      if (errno == EINTR) continue;
      system_error("write stream protocol");
    }
    if (count == 0) throw std::runtime_error("short stream protocol write");
    cursor += count;
    size -= static_cast<std::size_t>(count);
  }
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() { if (value_ >= 0) ::close(value_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : value_(other.value_) {
    other.value_ = -1;
  }
  int get() const { return value_; }

 private:
  int value_;
};

class MappedRegion {
 public:
  MappedRegion(int fd, std::size_t size) : size_(size) {
    if (fd < 0 || size == 0) return;
    data_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data_ == MAP_FAILED) system_error("mmap stream staging");
  }
  ~MappedRegion() {
    if (data_ && data_ != MAP_FAILED) ::munmap(data_, size_);
  }
  MappedRegion(const MappedRegion&) = delete;
  MappedRegion& operator=(const MappedRegion&) = delete;
  void* data() const { return data_ == MAP_FAILED ? nullptr : data_; }
  std::size_t size() const { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

class HostBuffer {
 public:
  HostBuffer(std::size_t size, bool cuda_host, bool initialize)
      : size_(size), cuda_host_(cuda_host) {
    if (size_ == 0) throw std::runtime_error("buffer size must be positive");
#ifdef PLLM_HAS_CUDA
    if (cuda_host_) {
      const auto status = cudaHostAlloc(&data_, size_, cudaHostAllocPortable);
      if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaHostAlloc: ") +
                                 cudaGetErrorString(status));
      }
    } else
#else
    if (cuda_host_) throw std::runtime_error("binary was built without CUDA");
#endif
    {
      if (posix_memalign(&data_, 4096, size_) != 0) {
        throw std::runtime_error("posix_memalign failed");
      }
    }
    if (initialize) std::memset(data_, 0, size_);
  }

  HostBuffer(void* external, std::size_t size)
      : data_(external), size_(size), external_(true) {
    if (!data_ || size_ == 0) {
      throw std::runtime_error("external buffer must be non-empty");
    }
  }

  ~HostBuffer() {
    if (!data_) return;
    if (external_) return;
#ifdef PLLM_HAS_CUDA
    if (cuda_host_) {
      cudaFreeHost(data_);
      return;
    }
#endif
    std::free(data_);
  }

  void* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
  bool cuda_host_ = false;
  bool external_ = false;
};

struct DeviceListDeleter {
  void operator()(ibv_device** value) const { if (value) ibv_free_device_list(value); }
};
struct ContextDeleter {
  void operator()(ibv_context* value) const { if (value) ibv_close_device(value); }
};
struct PdDeleter {
  void operator()(ibv_pd* value) const { if (value) ibv_dealloc_pd(value); }
};
struct CqDeleter {
  void operator()(ibv_cq* value) const { if (value) ibv_destroy_cq(value); }
};
struct QpDeleter {
  void operator()(ibv_qp* value) const { if (value) ibv_destroy_qp(value); }
};
struct MrDeleter {
  void operator()(ibv_mr* value) const { if (value) ibv_dereg_mr(value); }
};

class RdmaEndpoint {
 public:
  RdmaEndpoint(const Options& options, HostBuffer& buffer)
      : ib_port_(options.ib_port), gid_index_(options.gid_index), buffer_(buffer) {
    int count = 0;
    std::unique_ptr<ibv_device*, DeviceListDeleter> devices(
        ibv_get_device_list(&count));
    if (!devices || count == 0) throw std::runtime_error("no RDMA device found");
    ibv_device* selected = nullptr;
    for (int index = 0; index < count; ++index) {
      const std::string name = ibv_get_device_name(devices.get()[index]);
      if (options.device.empty() || options.device == name) {
        selected = devices.get()[index];
        device_name_ = name;
        break;
      }
    }
    if (!selected) throw std::runtime_error("requested RDMA device was not found");
    context_.reset(ibv_open_device(selected));
    if (!context_) throw std::runtime_error("ibv_open_device failed");
    if (ibv_query_port(context_.get(), ib_port_, &port_attr_) != 0) {
      throw std::runtime_error("ibv_query_port failed");
    }
    if (ibv_query_gid(context_.get(), ib_port_, gid_index_, &gid_) != 0) {
      throw std::runtime_error("ibv_query_gid failed");
    }
    ibv_device_attr device_attr{};
    if (ibv_query_device(context_.get(), &device_attr) != 0) {
      throw std::runtime_error("ibv_query_device failed");
    }
    rd_atomic_depth_ = static_cast<std::uint8_t>(std::min(
        {options.rd_atomic_depth,
         static_cast<std::size_t>(device_attr.max_qp_rd_atom),
         static_cast<std::size_t>(device_attr.max_qp_init_rd_atom),
         static_cast<std::size_t>(255)}));
    if (rd_atomic_depth_ == 0) {
      throw std::runtime_error("RDMA device does not support RDMA reads");
    }
    pd_.reset(ibv_alloc_pd(context_.get()));
    cq_.reset(ibv_create_cq(context_.get(), 64, nullptr, nullptr, 0));
    if (!pd_ || !cq_) throw std::runtime_error("failed to allocate PD/CQ");
    ibv_qp_init_attr init{};
    init.send_cq = cq_.get();
    init.recv_cq = cq_.get();
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 64;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = kHeaderBytes;
    qp_.reset(ibv_create_qp(pd_.get(), &init));
    if (!qp_) throw std::runtime_error("ibv_create_qp failed");
    max_inline_data_ = init.cap.max_inline_data;
    mr_.reset(ibv_reg_mr(pd_.get(), buffer.data(), buffer.size(),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                             IBV_ACCESS_REMOTE_READ));
    if (!mr_) throw std::runtime_error("ibv_reg_mr failed");
    psn_ = std::uniform_int_distribution<std::uint32_t>(0, 0xffffff)(rng_);
    to_init();
  }

  ConnectionWire local_wire() const {
    ConnectionWire wire{};
    wire.lid = htons(port_attr_.lid);
    wire.mtu = static_cast<std::uint8_t>(port_attr_.active_mtu);
    wire.gid_index = static_cast<std::uint8_t>(gid_index_);
    wire.rd_atomic_depth = rd_atomic_depth_;
    wire.qpn = htonl(qp_->qp_num);
    wire.psn = htonl(psn_);
    wire.rkey = htonl(mr_->rkey);
    wire.address = htobe64(reinterpret_cast<std::uintptr_t>(buffer_.data()));
    std::memcpy(wire.gid.data(), &gid_, wire.gid.size());
    return wire;
  }

  void connect(const ConnectionWire& remote_wire) {
    remote_ = remote_wire;
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = static_cast<ibv_mtu>(std::min(
        static_cast<int>(port_attr_.active_mtu),
        static_cast<int>(remote_.mtu)));
    attr.dest_qp_num = ntohl(remote_.qpn);
    attr.rq_psn = ntohl(remote_.psn);
    attr.max_dest_rd_atomic = rd_atomic_depth_;
    attr.min_rnr_timer = 12;
    attr.ah_attr.port_num = ib_port_;
    attr.ah_attr.dlid = ntohs(remote_.lid);
    union ibv_gid remote_gid{};
    std::memcpy(&remote_gid, remote_.gid.data(), remote_.gid.size());
    if (attr.ah_attr.dlid == 0 || !gid_is_zero(remote_gid)) {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.dgid = remote_gid;
      attr.ah_attr.grh.sgid_index = gid_index_;
      attr.ah_attr.grh.hop_limit = 1;
    }
    const int rtr_flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp_.get(), &attr, rtr_flags) != 0) {
      throw std::runtime_error("failed to move QP to RTR");
    }
    attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn_;
    attr.max_rd_atomic =
        std::min(rd_atomic_depth_, remote_.rd_atomic_depth);
    const int rts_flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp_.get(), &attr, rts_flags) != 0) {
      throw std::runtime_error("failed to move QP to RTS");
    }
  }

  void write(std::size_t local_offset, std::uint64_t remote_offset,
             std::size_t size) {
    transfer(IBV_WR_RDMA_WRITE, local_offset, remote_offset, size);
  }

  void read(std::size_t local_offset, std::uint64_t remote_offset,
            std::size_t size) {
    transfer(IBV_WR_RDMA_READ, local_offset, remote_offset, size);
  }

  void post_write(std::size_t local_offset, std::uint64_t remote_offset,
                  std::size_t size, bool signaled = true) {
    post(IBV_WR_RDMA_WRITE, local_offset, remote_offset, size, signaled);
  }

  void post_read(std::size_t local_offset, std::uint64_t remote_offset,
                 std::size_t size, bool signaled = true) {
    post(IBV_WR_RDMA_READ, local_offset, remote_offset, size, signaled);
  }

  void poll(std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) poll_completion();
  }

  const std::string& device_name() const { return device_name_; }
  std::uint8_t rd_atomic_depth() const { return rd_atomic_depth_; }
  std::uint32_t max_inline_data() const { return max_inline_data_; }

 private:
  void to_init() {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ib_port_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp_.get(), &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
      throw std::runtime_error("failed to move QP to INIT");
    }
  }

  void transfer(ibv_wr_opcode opcode, std::size_t local_offset,
                std::uint64_t remote_offset, std::size_t size) {
    if (local_offset + size > buffer_.size()) {
      throw std::runtime_error("local RDMA transfer exceeds staging buffer");
    }
    const std::uint64_t remote_address = be64toh(remote_.address);
    const std::uint32_t remote_key = ntohl(remote_.rkey);
    std::size_t completed = 0;
    while (completed < size) {
      const std::size_t length =
          std::min(kTransferChunk, size - completed);
      ibv_sge sge{};
      sge.addr = reinterpret_cast<std::uintptr_t>(buffer_.data()) +
                 local_offset + completed;
      sge.length = static_cast<std::uint32_t>(length);
      sge.lkey = mr_->lkey;
      ibv_send_wr wr{};
      wr.wr_id = completed + 1;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = opcode;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = remote_address + remote_offset + completed;
      wr.wr.rdma.rkey = remote_key;
      ibv_send_wr* bad = nullptr;
      if (ibv_post_send(qp_.get(), &wr, &bad) != 0) {
        throw std::runtime_error("ibv_post_send failed");
      }
      poll_completion();
      completed += length;
    }
  }

  void post(ibv_wr_opcode opcode, std::size_t local_offset,
            std::uint64_t remote_offset, std::size_t size, bool signaled) {
    if (size == 0 || size > kTransferChunk ||
        local_offset + size > buffer_.size()) {
      throw std::runtime_error("invalid pipelined RDMA transfer");
    }
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uintptr_t>(buffer_.data()) + local_offset;
    sge.length = static_cast<std::uint32_t>(size);
    sge.lkey = mr_->lkey;
    ibv_send_wr wr{};
    wr.wr_id = local_offset + 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = opcode;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    if (opcode == IBV_WR_RDMA_WRITE && size <= max_inline_data_) {
      wr.send_flags |= IBV_SEND_INLINE;
    }
    wr.wr.rdma.remote_addr = be64toh(remote_.address) + remote_offset;
    wr.wr.rdma.rkey = ntohl(remote_.rkey);
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp_.get(), &wr, &bad) != 0) {
      throw std::runtime_error("ibv_post_send pipelined transfer failed");
    }
  }

  void poll_completion() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
      ibv_wc completion{};
      const int count = ibv_poll_cq(cq_.get(), 1, &completion);
      if (count < 0) throw std::runtime_error("ibv_poll_cq failed");
      if (count == 0) continue;
      if (completion.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(std::string("RDMA completion failed: ") +
                                 ibv_wc_status_str(completion.status));
      }
      return;
    }
    throw std::runtime_error("RDMA completion timed out");
  }

  static bool gid_is_zero(const union ibv_gid& gid) {
    const std::array<std::uint8_t, 16> zero{};
    return std::memcmp(&gid, zero.data(), zero.size()) == 0;
  }

  std::uint8_t ib_port_;
  int gid_index_;
  HostBuffer& buffer_;
  std::string device_name_;
  ibv_port_attr port_attr_{};
  union ibv_gid gid_{};
  ConnectionWire remote_{};
  std::uint32_t psn_ = 0;
  std::uint8_t rd_atomic_depth_ = 1;
  std::uint32_t max_inline_data_ = 0;
  std::mt19937 rng_{std::random_device{}()};
  std::unique_ptr<ibv_context, ContextDeleter> context_;
  std::unique_ptr<ibv_pd, PdDeleter> pd_;
  std::unique_ptr<ibv_cq, CqDeleter> cq_;
  std::unique_ptr<ibv_qp, QpDeleter> qp_;
  std::unique_ptr<ibv_mr, MrDeleter> mr_;
};

std::uint64_t fnv1a(std::string_view value) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : value) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool secure_equal(std::string_view left, std::string_view right) {
  const std::size_t size = std::max(left.size(), right.size());
  std::size_t difference = left.size() ^ right.size();
  for (std::size_t index = 0; index < size; ++index) {
    const unsigned char lhs =
        index < left.size() ? static_cast<unsigned char>(left[index]) : 0;
    const unsigned char rhs =
        index < right.size() ? static_cast<unsigned char>(right[index]) : 0;
    difference |= static_cast<std::size_t>(lhs ^ rhs);
  }
  return difference == 0;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() {
      if (++index >= argc) {
        throw std::runtime_error("missing value for " + argument);
      }
      return std::string(argv[index]);
    };
    if (argument == "--server") options.server = true;
    else if (argument == "--client") options.client = value();
    else if (argument == "--port") options.port = static_cast<std::uint16_t>(std::stoul(value()));
    else if (argument == "--operation") {
      const auto operation = value();
      if (operation == "put") options.operation = Operation::Put;
      else if (operation == "get") options.operation = Operation::Get;
      else if (operation == "stream-get") options.operation = Operation::StreamGet;
      else throw std::runtime_error("operation must be put, get, or stream-get");
    } else if (argument == "--index") options.index_file = value();
    else if (argument == "--root") options.root = value();
    else if (argument == "--device") options.device = value();
    else if (argument == "--ib-port") options.ib_port = static_cast<std::uint8_t>(std::stoul(value()));
    else if (argument == "--gid-index") options.gid_index = std::stoi(value());
    else if (argument == "--allocator") options.allocator = value();
    else if (argument == "--token-file") options.token_file = value();
    else if (argument == "--insecure-no-auth") options.insecure_no_auth = true;
    else if (argument == "--pool-bytes") options.pool_bytes = std::stoull(value());
    else if (argument == "--slot-bytes") options.slot_bytes = std::stoull(value());
    else if (argument == "--queue-depth") options.queue_depth = std::stoull(value());
    else if (argument == "--rd-atomic-depth") options.rd_atomic_depth = std::stoull(value());
    else if (argument == "--stream-shm-fd") options.stream_shm_fd = std::stoi(value());
    else if (argument == "--stream-shm-bytes") options.stream_shm_bytes = std::stoull(value());
    else if (argument == "--help") {
      std::cout
          << "pllm-rdma-pool --server [--pool-bytes N] [--slot-bytes N]\n"
             "pllm-rdma-pool --client HOST --operation put|get --index FILE "
             "--root DIR [--queue-depth N] [--rd-atomic-depth N]\n"
             "pllm-rdma-pool --client HOST --operation stream-get "
             "--index FILE [--allocator cuda-host] "
             "[--stream-shm-fd FD --stream-shm-bytes N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.server == !options.client.empty()) {
    throw std::runtime_error("select exactly one of --server or --client");
  }
  if (!options.server && options.index_file.empty()) {
    throw std::runtime_error("client requires --index");
  }
  if (!options.server && options.operation != Operation::StreamGet &&
      options.root.empty()) {
    throw std::runtime_error("put/get client requires --root");
  }
  if (options.allocator != "aligned" && options.allocator != "cuda-host") {
    throw std::runtime_error("allocator must be aligned or cuda-host");
  }
  if (!options.token_file.empty()) {
    std::ifstream stream(options.token_file);
    std::getline(stream, options.token);
    if (!stream || options.token.empty()) {
      throw std::runtime_error("cannot read a non-empty token file");
    }
  }
  if (options.token.empty() && !options.insecure_no_auth) {
    throw std::runtime_error(
        "--token-file is required unless --insecure-no-auth is explicit");
  }
  if (options.slot_bytes < 4096 || options.pool_bytes < options.slot_bytes) {
    throw std::runtime_error("invalid pool or slot size");
  }
  if (options.queue_depth == 0 || options.queue_depth > 32) {
    throw std::runtime_error("queue depth must be within [1, 32]");
  }
  if (options.rd_atomic_depth == 0 || options.rd_atomic_depth > 255) {
    throw std::runtime_error("RDMA read atomic depth must be within [1, 255]");
  }
  if ((options.stream_shm_fd >= 0) != (options.stream_shm_bytes > 0)) {
    throw std::runtime_error(
        "--stream-shm-fd and --stream-shm-bytes must be provided together");
  }
  return options;
}

std::vector<IndexEntry> read_index(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open pool index");
  std::vector<IndexEntry> rows;
  std::set<std::uint64_t> slots;
  std::set<std::string> keys;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto first = line.find('\t');
    const auto second = first == std::string::npos
                            ? std::string::npos
                            : line.find('\t', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      throw std::runtime_error("invalid pool index row");
    }
    IndexEntry row;
    row.slot = std::stoull(line.substr(0, first));
    row.key = line.substr(first + 1, second - first - 1);
    row.size = std::stoull(line.substr(second + 1));
    if (row.key.empty() || row.key.is_absolute()) {
      throw std::runtime_error("invalid pool index key");
    }
    for (const auto& part : row.key) {
      if (part == "..") throw std::runtime_error("pool index key escapes root");
    }
    if (!slots.insert(row.slot).second || !keys.insert(row.key.string()).second) {
      throw std::runtime_error("duplicate pool index slot or key");
    }
    rows.push_back(row);
  }
  if (rows.empty()) throw std::runtime_error("pool index is empty");
  return rows;
}

FileDescriptor connect_tcp(const std::string& host, std::uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* raw = nullptr;
  const std::string service = std::to_string(port);
  if (getaddrinfo(host.c_str(), service.c_str(), &hints, &raw) != 0) {
    throw std::runtime_error("getaddrinfo failed");
  }
  std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw, freeaddrinfo);
  for (auto* address = addresses.get(); address; address = address->ai_next) {
    FileDescriptor socket_fd(
        ::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
    if (socket_fd.get() < 0) continue;
    if (::connect(socket_fd.get(), address->ai_addr, address->ai_addrlen) == 0) {
      return socket_fd;
    }
  }
  system_error("connect");
}

FileDescriptor listen_tcp(std::uint16_t port) {
  FileDescriptor socket_fd(::socket(AF_INET6, SOCK_STREAM, 0));
  if (socket_fd.get() < 0) system_error("socket");
  int enabled = 1;
  setsockopt(socket_fd.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
  int disabled = 0;
  setsockopt(socket_fd.get(), IPPROTO_IPV6, IPV6_V6ONLY, &disabled,
             sizeof(disabled));
  sockaddr_in6 address{};
  address.sin6_family = AF_INET6;
  address.sin6_addr = in6addr_any;
  address.sin6_port = htons(port);
  if (::bind(socket_fd.get(), reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
    system_error("bind");
  }
  if (::listen(socket_fd.get(), 16) != 0) system_error("listen");
  return socket_fd;
}

void exchange_connections(int socket_fd, RdmaEndpoint& endpoint) {
  const ConnectionWire local = endpoint.local_wire();
  send_all(socket_fd, &local, sizeof(local));
  ConnectionWire remote{};
  recv_all(socket_fd, &remote, sizeof(remote));
  endpoint.connect(remote);
}

PoolInfoWire make_pool_info(std::uint32_t status, std::uint64_t pool_bytes,
                            std::uint64_t slot_bytes,
                            std::uint64_t slot_count,
                            std::uint32_t message_size = 0) {
  return PoolInfoWire{htonl(status), htonl(message_size), htobe64(pool_bytes),
                      htobe64(slot_bytes), htobe64(slot_count)};
}

void authenticate_server(int socket_fd, const Options& options,
                         std::uint64_t slot_count) {
  AuthWire auth{};
  recv_all(socket_fd, &auth, sizeof(auth));
  if (ntohl(auth.magic) != kMagic || ntohs(auth.version) != kVersion) {
    throw std::runtime_error("invalid pool protocol header");
  }
  const std::uint32_t token_size = ntohl(auth.token_size);
  if (token_size > 4096) throw std::runtime_error("invalid token size");
  std::string token(token_size, '\0');
  if (token_size) recv_all(socket_fd, token.data(), token.size());
  if (!options.insecure_no_auth && !secure_equal(token, options.token)) {
    const std::string message = "authentication failed";
    const auto info = make_pool_info(1, 0, 0, 0, message.size());
    send_all(socket_fd, &info, sizeof(info));
    send_all(socket_fd, message.data(), message.size());
    throw std::runtime_error(message);
  }
  const auto info = make_pool_info(0, options.pool_bytes, options.slot_bytes,
                                   slot_count);
  send_all(socket_fd, &info, sizeof(info));
}

PoolInfoWire authenticate_client(int socket_fd, const Options& options) {
  const AuthWire auth{htonl(kMagic), htons(kVersion),
                      htons(static_cast<std::uint16_t>(options.operation)),
                      htonl(static_cast<std::uint32_t>(options.token.size()))};
  send_all(socket_fd, &auth, sizeof(auth));
  if (!options.token.empty()) {
    send_all(socket_fd, options.token.data(), options.token.size());
  }
  PoolInfoWire info{};
  recv_all(socket_fd, &info, sizeof(info));
  const std::uint32_t message_size = ntohl(info.message_size);
  std::string message(message_size, '\0');
  if (message_size) recv_all(socket_fd, message.data(), message.size());
  if (ntohl(info.status) != 0) {
    throw std::runtime_error("remote pool: " + message);
  }
  return info;
}

void read_file(const std::filesystem::path& path, HostBuffer& buffer,
               std::size_t offset, std::size_t size) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open source object: " + path.string());
  stream.read(static_cast<char*>(buffer.data()) + offset,
              static_cast<std::streamsize>(size));
  if (stream.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("short source object read");
  }
}

void write_file_atomic(const std::filesystem::path& path,
                       const HostBuffer& buffer, std::size_t offset,
                       std::size_t size) {
  std::filesystem::create_directories(path.parent_path());
  const auto temporary = path.string() + ".partial." + std::to_string(::getpid());
  try {
    FileDescriptor file(::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                               0600));
    if (file.get() < 0) system_error("open destination object");
    const auto* cursor = static_cast<const std::byte*>(buffer.data()) + offset;
    std::size_t remaining = size;
    while (remaining > 0) {
      const ssize_t written = ::write(file.get(), cursor, remaining);
      if (written < 0) {
        if (errno == EINTR) continue;
        system_error("write destination object");
      }
      if (written == 0) throw std::runtime_error("short destination write");
      cursor += written;
      remaining -= static_cast<std::size_t>(written);
    }
    if (::rename(temporary.c_str(), path.c_str()) != 0) {
      system_error("rename destination object");
    }
  } catch (...) {
    ::unlink(temporary.c_str());
    throw;
  }
}

SlotHeaderWire slot_header(const IndexEntry& entry, bool committed) {
  SlotHeaderWire header{};
  header.magic = htobe64(kSlotMagic);
  header.size = htobe64(entry.size);
  header.key_hash = htobe64(fnv1a(entry.key.generic_string()));
  header.slot = htobe64(entry.slot);
  header.committed = htobe64(committed ? 1 : 0);
  return header;
}

void validate_header(const SlotHeaderWire& header, const IndexEntry& entry) {
  if (be64toh(header.magic) != kSlotMagic ||
      be64toh(header.committed) != 1 ||
      be64toh(header.size) != entry.size ||
      be64toh(header.slot) != entry.slot ||
      be64toh(header.key_hash) != fnv1a(entry.key.generic_string())) {
    throw std::runtime_error("remote pool slot metadata mismatch for " +
                             entry.key.string());
  }
}

void finish_session(int socket_fd) {
  std::uint8_t done = 1;
  send_all(socket_fd, &done, sizeof(done));
  recv_all(socket_fd, &done, sizeof(done));
  if (done != 1) throw std::runtime_error("server did not commit pool session");
}

void write_stream_response(std::uint32_t status, const void* payload,
                           std::size_t payload_size,
                           std::string_view message = {},
                           bool write_payload = true) {
  const StreamResponseWire response{
      htonl(status), htobe64(payload_size),
      htonl(static_cast<std::uint32_t>(message.size()))};
  write_fd_all(STDOUT_FILENO, &response, sizeof(response));
  if (!message.empty()) {
    write_fd_all(STDOUT_FILENO, message.data(), message.size());
  }
  if (write_payload && payload_size > 0) {
    write_fd_all(STDOUT_FILENO, payload, payload_size);
  }
}

int run_stream_get(const Options& options, const std::vector<IndexEntry>& rows,
                   std::uint64_t slot_bytes, std::uint64_t slot_count,
                   int socket_fd) {
  std::unordered_map<std::string, const IndexEntry*> by_key;
  for (const auto& row : rows) {
    if (row.slot >= slot_count || row.size == 0 || row.size > slot_bytes) {
      throw std::runtime_error("index entry exceeds remote pool geometry");
    }
    by_key.emplace(row.key.generic_string(), &row);
  }
  const std::size_t data_region =
      static_cast<std::size_t>(slot_bytes) * options.queue_depth;
  const std::size_t header_region = kHeaderBytes * options.queue_depth;
  MappedRegion shared(options.stream_shm_fd, options.stream_shm_bytes);
  if (shared.data() != nullptr && shared.size() < data_region + header_region) {
    throw std::runtime_error("shared stream staging is smaller than one batch");
  }
  std::unique_ptr<HostBuffer> staging;
  if (shared.data() != nullptr) {
    staging = std::make_unique<HostBuffer>(
        shared.data(), data_region + header_region);
  } else {
    staging = std::make_unique<HostBuffer>(
        data_region + header_region, options.allocator == "cuda-host", false);
  }
  RdmaEndpoint endpoint(options, *staging);
  exchange_connections(socket_fd, endpoint);
  const std::uint64_t stride = kHeaderBytes + slot_bytes;
  std::uint64_t objects = 0;
  std::uint64_t payload_bytes = 0;

  while (true) {
    std::uint32_t encoded_size = 0;
    if (!read_fd_all(STDIN_FILENO, &encoded_size, sizeof(encoded_size), true)) {
      break;
    }
    const std::uint32_t first = ntohl(encoded_size);
    if (first == 0) break;
    std::uint32_t count = 1;
    std::vector<std::string> keys;
    if (first == UINT32_MAX) {
      std::uint32_t encoded_count = 0;
      read_fd_all(STDIN_FILENO, &encoded_count, sizeof(encoded_count));
      count = ntohl(encoded_count);
      if (count == 0 || count > options.queue_depth) {
        throw std::runtime_error("stream batch exceeds queue depth");
      }
    }
    keys.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
      std::uint32_t key_size = first;
      if (first == UINT32_MAX) {
        std::uint32_t encoded_key_size = 0;
        read_fd_all(STDIN_FILENO, &encoded_key_size, sizeof(encoded_key_size));
        key_size = ntohl(encoded_key_size);
      }
      if (key_size == 0 || key_size > 4096) {
        throw std::runtime_error("stream key exceeds 1-4096 bytes");
      }
      std::string key(key_size, '\0');
      read_fd_all(STDIN_FILENO, key.data(), key.size());
      keys.push_back(std::move(key));
    }
    try {
      std::vector<const IndexEntry*> resolved;
      resolved.reserve(keys.size());
      for (const auto& key : keys) {
        const auto found = by_key.find(key);
        resolved.push_back(found == by_key.end() ? nullptr : found->second);
      }
      std::vector<std::size_t> valid;
      for (std::size_t index = 0; index < resolved.size(); ++index) {
        if (resolved[index] != nullptr) valid.push_back(index);
      }
      for (std::size_t item = 0; item < valid.size(); ++item) {
        const std::size_t request_index = valid[item];
        const auto& row = *resolved[request_index];
        endpoint.post_read(data_region + request_index * kHeaderBytes,
                           row.slot * stride, sizeof(SlotHeaderWire), false);
      }
      for (std::size_t item = 0; item < valid.size(); ++item) {
        const std::size_t request_index = valid[item];
        const auto& row = *resolved[request_index];
        endpoint.post_read(request_index * slot_bytes,
                           row.slot * stride + kHeaderBytes,
                           static_cast<std::size_t>(row.size),
                           item + 1 == valid.size());
      }
      if (!valid.empty()) endpoint.poll(1);
      for (std::size_t item = 0; item < valid.size(); ++item) {
        const std::size_t request_index = valid[item];
        SlotHeaderWire header{};
        std::memcpy(&header,
                    static_cast<std::byte*>(staging->data()) + data_region +
                        request_index * kHeaderBytes,
                    sizeof(header));
        validate_header(header, *resolved[request_index]);
      }
      for (std::size_t request_index = 0; request_index < resolved.size();
           ++request_index) {
        const auto* row = resolved[request_index];
        if (row == nullptr) {
          write_stream_response(1, nullptr, 0,
                                "key is absent from the warm profile");
          continue;
        }
        if (shared.data() != nullptr) {
          // RDMA already landed in the shared mapping at request_index's
          // staging slot; no pipe or host memcpy is required.
        }
        write_stream_response(0,
                              static_cast<std::byte*>(staging->data()) +
                                  request_index * slot_bytes,
                              static_cast<std::size_t>(row->size), {},
                              shared.data() == nullptr);
        ++objects;
        payload_bytes += row->size;
      }
    } catch (const std::exception& error) {
      for (std::size_t index = 0; index < keys.size(); ++index) {
        write_stream_response(1, nullptr, 0, error.what());
      }
    }
  }
  finish_session(socket_fd);
  std::cerr << "{\"ready\":true,\"operation\":\"stream-get\","
            << "\"objects\":" << objects << ",\"bytes\":" << payload_bytes
            << ",\"persistent_qp\":true,\"remote_disk_io\":false,"
               "\"local_disk_io\":false,\"gpudirect_claimed\":false,"
            << "\"shared_staging\":"
            << (shared.data() != nullptr ? "true" : "false") << "}\n";
  return 0;
}

int run_server(const Options& options) {
  const std::uint64_t stride = kHeaderBytes + options.slot_bytes;
  const std::uint64_t slot_count = options.pool_bytes / stride;
  if (slot_count == 0) throw std::runtime_error("pool has no usable slots");
  HostBuffer pool(static_cast<std::size_t>(options.pool_bytes), false, true);
  auto listener = listen_tcp(options.port);
  std::cerr << "{\"ready\":true,\"backend\":\"one_sided_memory_pool\","
            << "\"protocol_version\":" << kVersion << ","
            << "\"pool_bytes\":" << options.pool_bytes
            << ",\"slot_bytes\":" << options.slot_bytes
            << ",\"slot_count\":" << slot_count
            << ",\"disk_on_data_path\":false,"
               "\"consistency_model\":\"phase_separated_epoch\"}\n";
  while (true) {
    sockaddr_storage address{};
    socklen_t length = sizeof(address);
    const int client_fd = ::accept(listener.get(),
                                   reinterpret_cast<sockaddr*>(&address),
                                   &length);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      system_error("accept");
    }
    std::thread([client_fd, &options, &pool, slot_count]() {
      FileDescriptor client(client_fd);
      try {
        authenticate_server(client.get(), options, slot_count);
        RdmaEndpoint endpoint(options, pool);
        exchange_connections(client.get(), endpoint);
        std::uint8_t done = 0;
        recv_all(client.get(), &done, sizeof(done));
        if (done != 1) {
          throw std::runtime_error("client session did not commit");
        }
        send_all(client.get(), &done, sizeof(done));
      } catch (const std::exception& error) {
        std::cerr << "{\"ready\":false,\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
      }
    }).detach();
  }
}

int run_client(const Options& options) {
  const auto rows = read_index(options.index_file);
  auto socket_fd = connect_tcp(options.client, options.port);
  const PoolInfoWire info = authenticate_client(socket_fd.get(), options);
  const std::uint64_t pool_bytes = be64toh(info.pool_bytes);
  const std::uint64_t slot_bytes = be64toh(info.slot_bytes);
  const std::uint64_t slot_count = be64toh(info.slot_count);
  if (options.operation == Operation::StreamGet) {
    return run_stream_get(
        options, rows, slot_bytes, slot_count, socket_fd.get());
  }
  for (const auto& row : rows) {
    if (row.slot >= slot_count || row.size == 0 || row.size > slot_bytes) {
      throw std::runtime_error("index entry exceeds remote pool geometry");
    }
  }
  const std::size_t data_region =
      static_cast<std::size_t>(slot_bytes) * options.queue_depth;
  const std::size_t header_region = kHeaderBytes * options.queue_depth;
  HostBuffer staging(data_region + header_region,
                     options.allocator == "cuda-host", false);
  RdmaEndpoint endpoint(options, staging);
  exchange_connections(socket_fd.get(), endpoint);
  const std::uint64_t stride = kHeaderBytes + slot_bytes;
  const auto started = std::chrono::steady_clock::now();
  std::uint64_t payload_bytes = 0;
  double rdma_seconds = 0.0;
  for (std::size_t begin = 0; begin < rows.size(); begin += options.queue_depth) {
    const std::size_t batch =
        std::min(options.queue_depth, rows.size() - begin);
    if (options.operation == Operation::Put) {
      // PLLM serializes PUT and GET epochs. RC ordering publishes each batch by
      // writing its compact commit headers after every payload write.
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        read_file(std::filesystem::path(options.root) / row.key, staging,
                  item * slot_bytes, static_cast<std::size_t>(row.size));
        const auto header = slot_header(row, true);
        const std::size_t header_offset = data_region + item * kHeaderBytes;
        std::memcpy(static_cast<std::byte*>(staging.data()) + header_offset,
                    &header, sizeof(header));
      }
      const auto rdma_started = std::chrono::steady_clock::now();
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        endpoint.post_write(item * slot_bytes,
                            row.slot * stride + kHeaderBytes,
                            static_cast<std::size_t>(row.size), false);
      }
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        const std::size_t header_offset = data_region + item * kHeaderBytes;
        endpoint.post_write(header_offset, row.slot * stride,
                            sizeof(SlotHeaderWire), item + 1 == batch);
      }
      endpoint.poll(1);
      rdma_seconds += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - rdma_started)
                          .count();
    } else {
      const auto rdma_started = std::chrono::steady_clock::now();
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        endpoint.post_read(data_region + item * kHeaderBytes,
                           row.slot * stride, sizeof(SlotHeaderWire), false);
      }
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        endpoint.post_read(item * slot_bytes,
                           row.slot * stride + kHeaderBytes,
                           static_cast<std::size_t>(row.size),
                           item + 1 == batch);
      }
      endpoint.poll(1);
      rdma_seconds += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - rdma_started)
                          .count();
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        SlotHeaderWire header{};
        std::memcpy(&header,
                    static_cast<std::byte*>(staging.data()) + data_region +
                        item * kHeaderBytes,
                    sizeof(header));
        validate_header(header, row);
      }
      for (std::size_t item = 0; item < batch; ++item) {
        const auto& row = rows[begin + item];
        write_file_atomic(std::filesystem::path(options.root) / row.key,
                          staging, item * slot_bytes,
                          static_cast<std::size_t>(row.size));
      }
    }
    for (std::size_t item = 0; item < batch; ++item) {
      payload_bytes += rows[begin + item].size;
    }
    const std::size_t completed = begin + batch;
    if (completed % 256 == 0 || completed == rows.size()) {
      std::cerr << completed << "/" << rows.size() << "\r";
    }
  }
  std::cerr << "\n";
  finish_session(socket_fd.get());
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  const double gbps = seconds > 0 ? payload_bytes * 8.0 / seconds / 1e9 : 0.0;
  const double rdma_gbps =
      rdma_seconds > 0 ? payload_bytes * 8.0 / rdma_seconds / 1e9 : 0.0;
  std::cout << "{\"ready\":true,\"operation\":\""
            << (options.operation == Operation::Put ? "put" : "get")
            << "\",\"objects\":" << rows.size()
            << ",\"bytes\":" << payload_bytes
            << ",\"seconds\":" << seconds
            << ",\"effective_gbps\":" << gbps
            << ",\"rdma_seconds\":" << rdma_seconds
            << ",\"rdma_effective_gbps\":" << rdma_gbps
            << ",\"pool_bytes\":" << pool_bytes
            << ",\"slot_bytes\":" << slot_bytes
            << ",\"queue_depth\":" << options.queue_depth
            << ",\"rd_atomic_depth\":"
            << static_cast<unsigned int>(endpoint.rd_atomic_depth())
            << ",\"max_inline_data\":" << endpoint.max_inline_data()
            << ",\"device\":\"" << json_escape(endpoint.device_name())
            << "\",\"persistent_qp\":true,\"one_sided\":true,"
               "\"remote_disk_io\":false,\"gpudirect_claimed\":false,"
               "\"consistency_model\":\"phase_separated_epoch\","
               "\"cqe_per_batch\":1}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    return options.server ? run_server(options) : run_client(options);
  } catch (const std::exception& error) {
    std::cerr << "{\"ready\":false,\"error\":\""
              << json_escape(error.what())
              << "\",\"gpudirect_claimed\":false}\n";
    return 2;
  }
}
