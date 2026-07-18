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

## One-sided remote memory pool

`pllm-rdma-pool` follows the persistent-pool design in
`doca_tensoroffloader_182` commit `4c6cf5a`: the remote host allocates and
registers one large MR, each client performs one QP handshake, and all profile
objects then move with one-sided RDMA reads/writes. The remote host performs no
filesystem I/O on the data path.

This is a research data plane, not a general-purpose concurrent object store.
PLLM's state machine separates offload and reload epochs, and sharded clients
own disjoint slot IDs. Under that invariant, RC ordering is sufficient: PUT
posts payload writes followed by 64-byte inline commit headers, while GET queues
header and payload reads together. A queue-depth-32 batch therefore posts up to
64 WRs but requests only one CQE. There is no payload checksum scan or extra
read-after-read round trip in the hot path. Full `.pllmex` SHA-256 and model
fingerprint validation is run at experiment boundaries.

RDMA READ credits are negotiated from both devices and default to 16 instead of
the single outstanding read used by the first prototype. `--rd-atomic-depth`
and `--queue-depth` are independent sweep variables. Client JSON reports:

- `rdma_seconds` / `rdma_effective_gbps`: verbs post and completion time,
  excluding local source reads and destination writes;
- `seconds` / `effective_gbps`: one worker's complete object phase;
- the sharded runner's `wall_seconds`: primary end-to-end metric including
  process startup, QP/MR setup and local I/O.

For Nemotron's 128-slot profile, 40 layers require 5,120 objects and about
14.8GiB. A 16GiB pool with 3MiB payload slots fits that working set; it does not
fit the complete 20,480-object export.

```bash
# ddst-71: mlx5_1, RoCEv2 GID 3
pllm-rdma-pool --server --port 17902 --device mlx5_1 --gid-index 3 \
  --pool-bytes 17179869184 --slot-bytes 3145728 \
  --token-file ~/.config/pllm/rdma-token

# ddst-75: build a 128-slot profile and offload it over one persistent QP
python scripts/rdma_memory_profile.py build \
  --root /mnt/ssd-storage/$USER/pllm-experts \
  --index results/eer-memory-profile.tsv --slots-per-layer 128
pllm-rdma-pool --client 192.168.70.71 --port 17902 --operation put \
  --index results/eer-memory-profile.tsv \
  --root /mnt/ssd-storage/$USER/pllm-experts \
  --allocator cuda-host --device mlx5_0 --gid-index 5 \
  --queue-depth 32 --rd-atomic-depth 16 \
  --token-file ~/.config/pllm/rdma-token
```

For four disjoint QPs, split the global profile without renumbering its slots
and run all clients concurrently:

```bash
python scripts/rdma_memory_profile.py split \
  --root /mnt/ssd-storage/$USER/pllm-experts \
  --index results/eer-memory-profile.tsv --shards 4
python scripts/run_rdma_memory_shards.py \
  --peer 192.168.70.71 --operation put \
  --root /mnt/ssd-storage/$USER/pllm-experts \
  --index results/eer-memory-profile-00.tsv \
  --index results/eer-memory-profile-01.tsv \
  --index results/eer-memory-profile-02.tsv \
  --index results/eer-memory-profile-03.tsv \
  --device mlx5_0 --gid-index 5 --queue-depth 32 \
  --rd-atomic-depth 16 --token-file ~/.config/pllm/rdma-token
```

The pool is volatile: a peer reboot invalidates every slot and all clients must
reconnect. On DGX Spark the path remains ConnectX-7 to registered host/UMA
staging memory; it does not claim GPUDirect RDMA.
