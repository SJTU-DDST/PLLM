#include <infiniband/verbs.h>

#ifdef PLLM_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
  std::string device;
  std::string allocator = "aligned";
  std::size_t bytes = 16ULL * 1024 * 1024;
  int iterations = 32;
};

std::string json_escape(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char ch : value) {
    if (ch == '\\' || ch == '"') result.push_back('\\');
    if (ch == '\n') {
      result += "\\n";
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::runtime_error("missing value for " + argument);
      return argv[index];
    };
    if (argument == "--device") options.device = value();
    else if (argument == "--allocator") options.allocator = value();
    else if (argument == "--bytes") options.bytes = std::stoull(value());
    else if (argument == "--iterations") options.iterations = std::stoi(value());
    else if (argument == "--help") {
      std::cout << "pllm-rdma-stage [--device mlx5_0] [--allocator aligned|cuda-host] "
                   "[--bytes N] [--iterations N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.bytes == 0 || options.bytes > 1024ULL * 1024 * 1024) {
    throw std::runtime_error("bytes must be between 1 and 1 GiB");
  }
  if (options.iterations < 1 || options.iterations > 10000) {
    throw std::runtime_error("iterations must be between 1 and 10000");
  }
  return options;
}

class HostBuffer {
 public:
  HostBuffer(std::size_t bytes, bool cuda_host) : bytes_(bytes), cuda_host_(cuda_host) {
#ifdef PLLM_HAS_CUDA
    if (cuda_host_) {
      const cudaError_t status = cudaHostAlloc(&data_, bytes_, cudaHostAllocPortable);
      if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaHostAlloc failed: ") + cudaGetErrorString(status));
      }
      return;
    }
#else
    if (cuda_host_) throw std::runtime_error("binary was built without CUDA runtime support");
#endif
    if (posix_memalign(&data_, 4096, bytes_) != 0) {
      data_ = nullptr;
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

  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  void* data() const { return data_; }
  std::size_t size() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  bool cuda_host_ = false;
};

std::string read_rate(const std::string& device) {
  const auto path = std::filesystem::path("/sys/class/infiniband") / device / "ports/1/rate";
  std::ifstream stream(path);
  std::string rate;
  std::getline(stream, rate);
  return rate;
}

struct DeviceListDeleter {
  void operator()(ibv_device** devices) const { if (devices) ibv_free_device_list(devices); }
};

struct ContextDeleter {
  void operator()(ibv_context* context) const { if (context) ibv_close_device(context); }
};

struct PdDeleter {
  void operator()(ibv_pd* pd) const { if (pd) ibv_dealloc_pd(pd); }
};

struct MrDeleter {
  void operator()(ibv_mr* mr) const { if (mr) ibv_dereg_mr(mr); }
};

int run(const Options& options) {
  int count = 0;
  std::unique_ptr<ibv_device*, DeviceListDeleter> devices(ibv_get_device_list(&count));
  if (!devices || count == 0) throw std::runtime_error("no RDMA device found");

  ibv_device* selected = nullptr;
  for (int index = 0; index < count; ++index) {
    const std::string name = ibv_get_device_name(devices.get()[index]);
    if (options.device.empty() || options.device == name) {
      selected = devices.get()[index];
      break;
    }
  }
  if (!selected) throw std::runtime_error("requested RDMA device was not found");
  const std::string device_name = ibv_get_device_name(selected);

  const bool cuda_host = options.allocator == "cuda-host";
  if (!cuda_host && options.allocator != "aligned") {
    throw std::runtime_error("allocator must be aligned or cuda-host");
  }
  HostBuffer source(options.bytes, cuda_host);
  HostBuffer destination(options.bytes, cuda_host);
  std::memset(source.data(), 0x5a, source.size());
  std::memset(destination.data(), 0, destination.size());

  std::unique_ptr<ibv_context, ContextDeleter> context(ibv_open_device(selected));
  if (!context) throw std::runtime_error("ibv_open_device failed");
  std::unique_ptr<ibv_pd, PdDeleter> pd(ibv_alloc_pd(context.get()));
  if (!pd) throw std::runtime_error("ibv_alloc_pd failed");

  const auto register_start = std::chrono::steady_clock::now();
  std::unique_ptr<ibv_mr, MrDeleter> mr(
      ibv_reg_mr(pd.get(), destination.data(), destination.size(),
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE));
  const auto register_end = std::chrono::steady_clock::now();
  if (!mr) throw std::runtime_error("ibv_reg_mr failed");

  const auto copy_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    std::memcpy(destination.data(), source.data(), options.bytes);
  }
  const auto copy_end = std::chrono::steady_clock::now();
  volatile unsigned char checksum = static_cast<unsigned char*>(destination.data())[options.bytes - 1];
  (void)checksum;

  const double registration_us = std::chrono::duration<double, std::micro>(register_end - register_start).count();
  const double copy_seconds = std::chrono::duration<double>(copy_end - copy_start).count();
  const double gbps = static_cast<double>(options.bytes) * options.iterations * 8.0 / copy_seconds / 1e9;

  std::cout << std::fixed << std::setprecision(3)
            << "{\"transport\":\"host_staged\",\"device\":\"" << json_escape(device_name)
            << "\",\"link_rate\":\"" << json_escape(read_rate(device_name))
            << "\",\"allocator\":\"" << json_escape(options.allocator)
            << "\",\"bytes\":" << options.bytes
            << ",\"iterations\":" << options.iterations
            << ",\"mr_registered\":true,\"registration_us\":" << registration_us
            << ",\"staging_bandwidth_gbps\":" << gbps
            << ",\"gpu_allocated\":false,\"gpudirect_claimed\":false}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "{\"transport\":\"host_staged\",\"ready\":false,\"error\":\""
              << json_escape(error.what())
              << "\",\"gpu_allocated\":false,\"gpudirect_claimed\":false}\n";
    return 2;
  }
}
