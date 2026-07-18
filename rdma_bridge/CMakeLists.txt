cmake_minimum_required(VERSION 3.20)
project(pllm_rdma_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_package(CUDAToolkit QUIET)
find_package(Threads REQUIRED)

add_executable(pllm-rdma-stage src/main.cpp)
target_include_directories(pllm-rdma-stage PRIVATE ${IBVERBS_INCLUDE_DIR})
target_link_libraries(pllm-rdma-stage PRIVATE ${IBVERBS_LIBRARY})
target_compile_options(pllm-rdma-stage PRIVATE -Wall -Wextra -Wpedantic)

if(CUDAToolkit_FOUND)
  target_compile_definitions(pllm-rdma-stage PRIVATE PLLM_HAS_CUDA=1)
  target_link_libraries(pllm-rdma-stage PRIVATE CUDA::cudart)
endif()

add_executable(pllm-rdma-store src/store.cpp)
target_include_directories(pllm-rdma-store PRIVATE ${IBVERBS_INCLUDE_DIR})
target_link_libraries(pllm-rdma-store PRIVATE ${IBVERBS_LIBRARY})
target_compile_options(pllm-rdma-store PRIVATE -Wall -Wextra -Wpedantic)

if(CUDAToolkit_FOUND)
  target_compile_definitions(pllm-rdma-store PRIVATE PLLM_HAS_CUDA=1)
  target_link_libraries(pllm-rdma-store PRIVATE CUDA::cudart)
endif()

add_executable(pllm-rdma-pool src/pool.cpp)
target_include_directories(pllm-rdma-pool PRIVATE ${IBVERBS_INCLUDE_DIR})
target_link_libraries(pllm-rdma-pool PRIVATE ${IBVERBS_LIBRARY} Threads::Threads)
target_compile_options(pllm-rdma-pool PRIVATE -Wall -Wextra -Wpedantic)

if(CUDAToolkit_FOUND)
  target_compile_definitions(pllm-rdma-pool PRIVATE PLLM_HAS_CUDA=1)
  target_link_libraries(pllm-rdma-pool PRIVATE CUDA::cudart)
endif()
