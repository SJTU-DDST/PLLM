#include <infiniband/verbs.h>

#ifdef PLLM_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <arpa/inet.h>
#include <endian.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x504c4c4d;
constexpr std::uint16_t kVersion = 1;
constexpr std::uint64_t kMaxObjectBytes = 128ULL * 1024 * 1024;
constexpr std::size_t kWriteChunk = 8ULL * 1024 * 1024;

enum class Operation : std::uint16_t { Get = 1, Put = 2 };

struct Options {
  bool server = false;
  bool once = false;
  std::string client;
  std::string root;
  std::string file;
  std::string key;
  std::string device;
  std::string allocator = "aligned";
  std::string token;
  std::string token_file;
  bool insecure_no_auth = false;
  Operation operation = Operation::Get;
  std::uint16_t port = 17900;
  std::uint8_t ib_port = 1;
  int gid_index = 0;
};

#pragma pack(push, 1)
struct RequestWire {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t operation;
  std::uint32_t token_size;
  std::uint32_t key_size;
  std::uint64_t object_size;
};

struct StatusWire {
  std::uint32_t code;
  std::uint32_t message_size;
  std::uint64_t object_size;
};

struct ConnectionWire {
  std::uint16_t lid;
  std::uint8_t mtu;
  std::uint8_t gid_index;
  std::uint32_t qpn;
  std::uint32_t psn;
  std::uint32_t rkey;
  std::uint64_t address;
  std::array<std::uint8_t, 16> gid;
};
#pragma pack(pop)

std::string json_escape(std::string_view value) {
  std::string output;
  for (char ch : value) {
    if (ch == '\\' || ch == '"') output.push_back('\\');
    if (ch == '\n') output += "\\n";
    else output.push_back(ch);
  }
  return output;
}

[[noreturn]] void system_error(const std::string& operation) {
  throw std::runtime_error(operation + ": " + std::strerror(errno));
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

class HostBuffer {
 public:
  HostBuffer(std::size_t size, bool cuda_host) : size_(size), cuda_host_(cuda_host) {
    if (size_ == 0 || size_ > kMaxObjectBytes) {
      throw std::runtime_error("expert object size is outside the allowed range");
    }
#ifdef PLLM_HAS_CUDA
    if (cuda_host_) {
      const auto status = cudaHostAlloc(&data_, size_, cudaHostAllocPortable);
      if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaHostAlloc: ") + cudaGetErrorString(status));
      }
      return;
    }
#else
    if (cuda_host_) throw std::runtime_error("binary was built without CUDA");
#endif
    if (posix_memalign(&data_, 4096, size_) != 0) {
      throw std::runtime_error("posix_memalign failed");
    }
  }

  ~HostBuffer() {
    if (!data_) return;
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
    std::unique_ptr<ibv_device*, DeviceListDeleter> devices(ibv_get_device_list(&count));
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
    pd_.reset(ibv_alloc_pd(context_.get()));
    cq_.reset(ibv_create_cq(context_.get(), 32, nullptr, nullptr, 0));
    if (!pd_ || !cq_) throw std::runtime_error("failed to allocate PD/CQ");
    ibv_qp_init_attr init{};
    init.send_cq = cq_.get();
    init.recv_cq = cq_.get();
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 32;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    qp_.reset(ibv_create_qp(pd_.get(), &init));
    if (!qp_) throw std::runtime_error("ibv_create_qp failed");
    mr_.reset(ibv_reg_mr(pd_.get(), buffer.data(), buffer.size(),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
    if (!mr_) throw std::runtime_error("ibv_reg_mr failed");
    psn_ = std::uniform_int_distribution<std::uint32_t>(0, 0xffffff)(rng_);
    to_init();
  }

  ConnectionWire local_wire() const {
    ConnectionWire wire{};
    wire.lid = htons(port_attr_.lid);
    wire.mtu = static_cast<std::uint8_t>(port_attr_.active_mtu);
    wire.gid_index = static_cast<std::uint8_t>(gid_index_);
    wire.qpn = htonl(qp_->qp_num);
    wire.psn = htonl(psn_);
    wire.rkey = htonl(mr_->rkey);
    wire.address = htobe64(reinterpret_cast<std::uintptr_t>(buffer_.data()));
    std::memcpy(wire.gid.data(), &gid_, wire.gid.size());
    return wire;
  }

  void connect(const ConnectionWire& remote_wire) {
    remote_ = remote_wire;
    const std::uint32_t remote_qpn = ntohl(remote_.qpn);
    const std::uint32_t remote_psn = ntohl(remote_.psn);
    const std::uint16_t remote_lid = ntohs(remote_.lid);
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = static_cast<ibv_mtu>(std::min(
        static_cast<int>(port_attr_.active_mtu), static_cast<int>(remote_.mtu)));
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.port_num = ib_port_;
    attr.ah_attr.dlid = remote_lid;
    attr.ah_attr.sl = 0;
    union ibv_gid remote_gid{};
    std::memcpy(&remote_gid, remote_.gid.data(), remote_.gid.size());
    if (remote_lid == 0 || !gid_is_zero(remote_gid)) {
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
    attr.max_rd_atomic = 1;
    const int rts_flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp_.get(), &attr, rts_flags) != 0) {
      throw std::runtime_error("failed to move QP to RTS");
    }
  }

  void write_to_remote() {
    const std::uint64_t remote_address = be64toh(remote_.address);
    const std::uint32_t remote_key = ntohl(remote_.rkey);
    std::size_t offset = 0;
    while (offset < buffer_.size()) {
      const std::size_t length = std::min(kWriteChunk, buffer_.size() - offset);
      ibv_sge sge{};
      sge.addr = reinterpret_cast<std::uintptr_t>(buffer_.data()) + offset;
      sge.length = static_cast<std::uint32_t>(length);
      sge.lkey = mr_->lkey;
      ibv_send_wr wr{};
      wr.wr_id = offset + 1;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = remote_address + offset;
      wr.wr.rdma.rkey = remote_key;
      ibv_send_wr* bad = nullptr;
      if (ibv_post_send(qp_.get(), &wr, &bad) != 0) {
        throw std::runtime_error("ibv_post_send RDMA_WRITE failed");
      }
      poll_completion();
      offset += length;
    }
  }

  const std::string& device_name() const { return device_name_; }

 private:
  void to_init() {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ib_port_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp_.get(), &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
      throw std::runtime_error("failed to move QP to INIT");
    }
  }

  void poll_completion() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
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
    throw std::runtime_error("RDMA write completion timed out");
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
  std::mt19937 rng_{std::random_device{}()};
  std::unique_ptr<ibv_context, ContextDeleter> context_;
  std::unique_ptr<ibv_pd, PdDeleter> pd_;
  std::unique_ptr<ibv_cq, CqDeleter> cq_;
  std::unique_ptr<ibv_qp, QpDeleter> qp_;
  std::unique_ptr<ibv_mr, MrDeleter> mr_;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() {
      if (++index >= argc) throw std::runtime_error("missing value for " + argument);
      return std::string(argv[index]);
    };
    if (argument == "--server") { options.server = true; options.root = value(); }
    else if (argument == "--client") options.client = value();
    else if (argument == "--port") options.port = static_cast<std::uint16_t>(std::stoul(value()));
    else if (argument == "--operation") {
      const auto operation = value();
      if (operation == "get") options.operation = Operation::Get;
      else if (operation == "put") options.operation = Operation::Put;
      else throw std::runtime_error("operation must be get or put");
    } else if (argument == "--key") options.key = value();
    else if (argument == "--file") options.file = value();
    else if (argument == "--device") options.device = value();
    else if (argument == "--ib-port") options.ib_port = static_cast<std::uint8_t>(std::stoul(value()));
    else if (argument == "--gid-index") options.gid_index = std::stoi(value());
    else if (argument == "--allocator") options.allocator = value();
    else if (argument == "--token-file") options.token_file = value();
    else if (argument == "--insecure-no-auth") options.insecure_no_auth = true;
    else if (argument == "--once") options.once = true;
    else if (argument == "--help") {
      std::cout << "pllm-rdma-store --server ROOT [--port 17900] [--once]\n"
                   "pllm-rdma-store --client HOST --operation get|put --key KEY --file PATH\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.server == !options.client.empty()) {
    throw std::runtime_error("select exactly one of --server or --client");
  }
  if (!options.server && (options.key.empty() || options.file.empty())) {
    throw std::runtime_error("client requires --key and --file");
  }
  if (options.allocator != "aligned" && options.allocator != "cuda-host") {
    throw std::runtime_error("allocator must be aligned or cuda-host");
  }
  if (!options.token_file.empty()) {
    std::ifstream stream(options.token_file);
    std::getline(stream, options.token);
    if (!stream || options.token.empty()) throw std::runtime_error("cannot read a non-empty token file");
  }
  if (options.token.empty() && !options.insecure_no_auth) {
    throw std::runtime_error("--token-file is required unless --insecure-no-auth is explicit");
  }
  return options;
}

bool secure_equal(std::string_view left, std::string_view right) {
  const std::size_t size = std::max(left.size(), right.size());
  std::size_t difference = left.size() ^ right.size();
  for (std::size_t index = 0; index < size; ++index) {
    const unsigned char lhs = index < left.size() ? static_cast<unsigned char>(left[index]) : 0;
    const unsigned char rhs = index < right.size() ? static_cast<unsigned char>(right[index]) : 0;
    difference |= static_cast<std::size_t>(lhs ^ rhs);
  }
  return difference == 0;
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
    FileDescriptor socket_fd(::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
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
  setsockopt(socket_fd.get(), IPPROTO_IPV6, IPV6_V6ONLY, &disabled, sizeof(disabled));
  sockaddr_in6 address{};
  address.sin6_family = AF_INET6;
  address.sin6_addr = in6addr_any;
  address.sin6_port = htons(port);
  if (::bind(socket_fd.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) system_error("bind");
  if (::listen(socket_fd.get(), 16) != 0) system_error("listen");
  return socket_fd;
}

std::filesystem::path safe_path(const std::filesystem::path& root, const std::string& key) {
  const std::filesystem::path relative(key);
  if (relative.empty() || relative.is_absolute()) throw std::runtime_error("invalid object key");
  for (const auto& part : relative) {
    if (part == "..") throw std::runtime_error("object key escapes the store root");
  }
  return (root / relative).lexically_normal();
}

void read_file(const std::filesystem::path& path, HostBuffer& buffer) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open source object: " + path.string());
  stream.read(static_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
  if (stream.gcount() != static_cast<std::streamsize>(buffer.size())) throw std::runtime_error("short object read");
}

void write_file_atomic(const std::filesystem::path& path, const HostBuffer& buffer) {
  std::filesystem::create_directories(path.parent_path());
  const auto temporary = path.string() + ".partial." + std::to_string(::getpid());
  try {
    FileDescriptor file(::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600));
    if (file.get() < 0) system_error("open destination object");
    const auto* cursor = static_cast<const std::byte*>(buffer.data());
    std::size_t remaining = buffer.size();
    while (remaining > 0) {
      const ssize_t written = ::write(file.get(), cursor, remaining);
      if (written < 0) {
        if (errno == EINTR) continue;
        system_error("write destination object");
      }
      if (written == 0) throw std::runtime_error("short destination object write");
      cursor += written;
      remaining -= static_cast<std::size_t>(written);
    }
    if (::fsync(file.get()) != 0) system_error("fsync destination object");
    if (::rename(temporary.c_str(), path.c_str()) != 0) system_error("rename destination object");
    FileDescriptor directory(::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY));
    if (directory.get() < 0) system_error("open destination directory");
    if (::fsync(directory.get()) != 0) system_error("fsync destination directory");
  } catch (...) {
    ::unlink(temporary.c_str());
    throw;
  }
}

void exchange_connections(int socket_fd, RdmaEndpoint& endpoint) {
  const ConnectionWire local = endpoint.local_wire();
  send_all(socket_fd, &local, sizeof(local));
  ConnectionWire remote{};
  recv_all(socket_fd, &remote, sizeof(remote));
  endpoint.connect(remote);
}

void send_status(int socket_fd, std::uint32_t code, std::uint64_t size, const std::string& message = {}) {
  const StatusWire wire{htonl(code), htonl(static_cast<std::uint32_t>(message.size())), htobe64(size)};
  send_all(socket_fd, &wire, sizeof(wire));
  if (!message.empty()) send_all(socket_fd, message.data(), message.size());
}

std::pair<std::uint64_t, std::string> recv_status(int socket_fd) {
  StatusWire wire{};
  recv_all(socket_fd, &wire, sizeof(wire));
  const auto code = ntohl(wire.code);
  const auto message_size = ntohl(wire.message_size);
  std::string message(message_size, '\0');
  if (message_size) recv_all(socket_fd, message.data(), message.size());
  if (code != 0) throw std::runtime_error("remote store: " + message);
  return {be64toh(wire.object_size), message};
}

int run_client(const Options& options) {
  const bool put = options.operation == Operation::Put;
  std::uint64_t size = 0;
  if (put) {
    size = std::filesystem::file_size(options.file);
    if (size == 0 || size > kMaxObjectBytes) throw std::runtime_error("source object size is invalid");
  }
  auto socket_fd = connect_tcp(options.client, options.port);
  RequestWire request{htonl(kMagic), htons(kVersion), htons(static_cast<std::uint16_t>(options.operation)),
                      htonl(static_cast<std::uint32_t>(options.token.size())),
                      htonl(static_cast<std::uint32_t>(options.key.size())), htobe64(size)};
  send_all(socket_fd.get(), &request, sizeof(request));
  if (!options.token.empty()) send_all(socket_fd.get(), options.token.data(), options.token.size());
  send_all(socket_fd.get(), options.key.data(), options.key.size());
  size = recv_status(socket_fd.get()).first;
  HostBuffer buffer(static_cast<std::size_t>(size), options.allocator == "cuda-host");
  if (put) read_file(options.file, buffer);
  RdmaEndpoint endpoint(options, buffer);
  exchange_connections(socket_fd.get(), endpoint);
  std::uint8_t done = 1;
  if (put) {
    endpoint.write_to_remote();
    send_all(socket_fd.get(), &done, sizeof(done));
    recv_all(socket_fd.get(), &done, sizeof(done));
  } else {
    recv_all(socket_fd.get(), &done, sizeof(done));
  }
  if (done != 1) throw std::runtime_error("remote transfer did not commit");
  if (!put) write_file_atomic(options.file, buffer);
  std::cout << "{\"ready\":true,\"operation\":\"" << (put ? "put" : "get")
            << "\",\"bytes\":" << size << ",\"transport\":\"rdma_write_host_staged\","
            << "\"device\":\"" << json_escape(endpoint.device_name())
            << "\",\"gpudirect_claimed\":false}\n";
  return 0;
}

void handle_server_client(int socket_fd, const Options& options) {
  RequestWire request{};
  recv_all(socket_fd, &request, sizeof(request));
  if (ntohl(request.magic) != kMagic || ntohs(request.version) != kVersion) {
    throw std::runtime_error("invalid PLLM RDMA protocol header");
  }
  const auto token_size = ntohl(request.token_size);
  if (token_size > 4096) throw std::runtime_error("invalid authentication token size");
  std::string token(token_size, '\0');
  if (token_size) recv_all(socket_fd, token.data(), token.size());
  if (!options.insecure_no_auth && !secure_equal(token, options.token)) {
    throw std::runtime_error("authentication failed");
  }
  const auto key_size = ntohl(request.key_size);
  if (key_size == 0 || key_size > 4096) throw std::runtime_error("invalid object key size");
  std::string key(key_size, '\0');
  recv_all(socket_fd, key.data(), key.size());
  const auto path = safe_path(options.root, key);
  const auto operation = static_cast<Operation>(ntohs(request.operation));
  std::uint64_t size = be64toh(request.object_size);
  if (operation == Operation::Get) {
    size = std::filesystem::file_size(path);
  }
  if (size == 0 || size > kMaxObjectBytes) throw std::runtime_error("object size is invalid");
  HostBuffer buffer(static_cast<std::size_t>(size), options.allocator == "cuda-host");
  if (operation == Operation::Get) read_file(path, buffer);
  RdmaEndpoint endpoint(options, buffer);
  send_status(socket_fd, 0, size);
  exchange_connections(socket_fd, endpoint);
  std::uint8_t done = 1;
  if (operation == Operation::Get) {
    endpoint.write_to_remote();
    send_all(socket_fd, &done, sizeof(done));
  } else if (operation == Operation::Put) {
    recv_all(socket_fd, &done, sizeof(done));
    if (done != 1) throw std::runtime_error("sender did not complete RDMA write");
    write_file_atomic(path, buffer);
    send_all(socket_fd, &done, sizeof(done));
  } else {
    throw std::runtime_error("unknown object operation");
  }
}

int run_server(const Options& options) {
  std::filesystem::create_directories(options.root);
  auto listener = listen_tcp(options.port);
  do {
    sockaddr_storage address{};
    socklen_t length = sizeof(address);
    FileDescriptor client(::accept(listener.get(), reinterpret_cast<sockaddr*>(&address), &length));
    if (client.get() < 0) {
      if (errno == EINTR) continue;
      system_error("accept");
    }
    try {
      handle_server_client(client.get(), options);
    } catch (const std::exception& error) {
      try { send_status(client.get(), 1, 0, error.what()); } catch (...) {}
      std::cerr << "{\"ready\":false,\"error\":\"" << json_escape(error.what()) << "\"}\n";
    }
  } while (!options.once);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    return options.server ? run_server(options) : run_client(options);
  } catch (const std::exception& error) {
    std::cerr << "{\"ready\":false,\"error\":\"" << json_escape(error.what())
              << "\",\"gpudirect_claimed\":false}\n";
    return 2;
  }
}
