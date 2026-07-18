# PLLM host-staged RDMA bridge

This benchmark validates the DGX Spark compatible staging path without claiming
GPUDirect RDMA support. It allocates a host buffer, registers it as an verbs MR,
and measures the copy into that registered buffer. `cuda-host` uses
`cudaHostAlloc`; `aligned` performs the current no-GPU acceptance test.

The ownership boundaries follow `/home/cong/rdmapp`: device, PD and MR are
independent RAII resources, while connection setup and metrics stay outside the
resource wrappers. The original repository is not modified or linked.

```bash
cmake -S rdma_bridge -B rdma_bridge/build -DCMAKE_BUILD_TYPE=Release
cmake --build rdma_bridge/build -j
rdma_bridge/build/pllm-rdma-stage --allocator aligned
```

For a real two-host link measurement, run `scripts/rdma_benchmark.py --server`
on the peer and `scripts/rdma_benchmark.py --peer <address>` on the PLLM host.
The script keeps MR staging bandwidth and network RDMA write bandwidth as
separate metrics.

## Expert object data plane

`pllm-rdma-store` is the non-benchmark data path. The remote host exposes a
directory containing checksummed `.pllmex` runtime-expert objects. A compute
host registers an aligned or `cudaHostAlloc` destination buffer, exchanges its
MR over the TCP control channel, and receives the object with RC RDMA write.
The Python tier verifies the package SHA-256 and model fingerprint before it is
eligible for a Marlin physical slot.

```bash
# Warm-source host
rdma_bridge/build/pllm-rdma-store \
  --server /mnt/ssd-storage/$USER/pllm-experts --port 17900 --device mlx5_0 \
  --token-file ~/.config/pllm/rdma-token

# DGX Spark compatible host-staged fetch
rdma_bridge/build/pllm-rdma-store \
  --client 192.168.70.71 --port 17900 --operation get \
  --key layer-001/expert-0000.pllmex --file /tmp/expert.pllmex \
  --allocator cuda-host --device mlx5_0 \
  --token-file ~/.config/pllm/rdma-token
```

The bridge also supports `--operation put`. A PUT ACK is sent only after the
remote temporary file and parent directory have been `fsync`ed around the
atomic rename. It never claims GPUDirect RDMA: the registered destination is
host memory; the standalone bridge first commits a local SSD cache object, and
Python then validates it before copying it into the physical expert slot.
