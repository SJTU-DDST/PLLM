# Pause-MoE Locality and Residency Experiment

Date: 2026-07-19

All model serving and GPU measurements in this report ran on the local machine.
No remote host was used. The expert cache path is derived from the effective
user rather than a hard-coded account name.

## Executive result

Evicting experts after a token-boundary pause did not improve decode speed on
this workload. Pause-only resumed at essentially baseline speed. Reducing the
active or physical expert cache increased SSD reloads and made decode slower.

- 384-slot resident baseline: 206.8 ms mean TPOT.
- Pause-only at 384 slots: 210.0 ms post-pause TPOT, 1.016x baseline.
- Logical eviction to 368 slots: 234.2 ms post-pause TPOT, 1.133x baseline.
- Logical eviction to 352 slots: 318.9 ms post-pause TPOT, 1.542x baseline.
- Logical eviction to 320 slots: 386.1 ms post-pause TPOT, 1.867x baseline.
- Static physical 320 slots: 381.9 ms mean TPOT, 1.847x baseline, while
  reclaiming 7.34 GiB of measured GPU memory.

The 320-slot logical and physical slowdowns agree closely. This indicates that
the dominant cost is the smaller expert working set and its reload misses, not
the logical capacity control operation.

## Route locality

The route capture contains 60 successful requests, 1,033 decode tokens, and
909,040 expert accesses from the MQA, NQA, and TQA samples.

Recent-token reuse is real but not strong enough for aggressive eviction:

| Decode history | Accesses covered by prior experts | Mean working set/layer |
| ---: | ---: | ---: |
| 1 token | 33.7% | 22.0 |
| 8 tokens | 63.7% | 97.4 |
| 32 tokens | 82.9% | 237.0 |
| 64 tokens | 90.7% | 331.2 |
| 128 tokens | 95.5% | 401.3 |

Window-LFU replay of the captured routes gives this trade-off relative to 512
resident experts per layer:

| Slots/layer | Byte hit rate | Projected reclaim | Estimated slowdown |
| ---: | ---: | ---: | ---: |
| 504 | 99.63% | 0.92 GiB | 1.24x |
| 496 | 99.25% | 1.85 GiB | 1.49x |
| 480 | 98.48% | 3.69 GiB | 2.00x |
| 448 | 96.83% | 7.38 GiB | 3.09x |
| 384 | 92.87% | 14.77 GiB | 5.71x |

The replay slowdown is an estimate using the measured runtime miss model. The
memory values in this table are projections, not live reclamation claims.

## Live pause cycles

The live test used two warmup requests followed by three rotated rounds for
each arm. All 15 measured requests completed. The logical capacity transition
preserved the live KV/Mamba state allocation fingerprint and copied zero state
bytes.

| Arm | Action/gap | Post-pause TPOT | Slowdown | Measured reclaim |
| --- | ---: | ---: | ---: | ---: |
| Baseline 384 | 0 ms | 206.8 ms | 1.000x | 0 GiB |
| Pause only | 199.4 ms | 210.0 ms | 1.016x | 0 GiB |
| Evict to 368 | 303.9 ms | 234.2 ms | 1.133x | 0 GiB |
| Evict to 352 | 308.4 ms | 318.9 ms | 1.542x | 0 GiB |
| Evict to 320 | 290.9 ms | 386.1 ms | 1.867x | 0 GiB |

Logical eviction intentionally leaves physical weight tensors allocated. Its
zero measured reclaim is expected and is shown explicitly so projected and
measured memory are not mixed.

## Physical profile check

The static 384- and 320-slot services used the same 7.56 GiB KV/state cache.

| Physical slots/layer | Model load memory | Measured GPU memory | Mean TPOT |
| ---: | ---: | ---: | ---: |
| 384 | 54.93 GiB | 64.59 GiB | 206.8 ms |
| 320 | 47.59 GiB | 57.25 GiB | 381.9 ms |

Physical 320 slots reclaimed 7,520 MiB (7.34 GiB), matching the 7.38 GiB
projection within measurement precision, but increased TPOT by 84.7%.

## Physical online resize boundary

The current vLLM 0.25.1 sleep-mode allocator places model weights in a private
CuMem pool. Online physical resize allocates replacement expert parameters from
a different allocator context, so released private-pool blocks are not reused.
The 384-to-368 online physical resize therefore reached an allocation peak and
failed with CUDA OOM. Re-entering that pool from the control thread was also
tested and rejected after an illegal-address fault; that unsafe change was
removed.

The supported live path in this experiment is consequently logical expert
eviction. Physical resize remains opt-in and should not be enabled until weight
rebuild is moved onto vLLM's owning worker/allocator context.

## Recommendation

For speed after a pause, keep the existing expert residency and use pause-only.
Only shrink the expert set when freeing VRAM is more important than decode
latency. On this GPU and workload, 320 physical slots trade about 7.34 GiB for
an 84.7% TPOT increase. A conservative eviction point near 496-504 slots is the
only replayed region with modest slowdown, but those profiles do not fit this
EER startup envelope and still need a worker-owned physical resize path.

## Artifacts

- [Live pause and logical eviction](live_pause_moe_performance.png)
- [Static physical residency comparison](physical_moe_profiles.png)
- [Hit rate versus projected reclaim](hit_rate_vs_reclaim.png)
- [Miss cost versus projected reclaim](miss_cost_vs_reclaim.png)
- [Token-history locality](token_history_locality.png)
- [Live summary CSV](live_pause_moe_summary.csv)
- [Physical summary CSV](physical_moe_profiles.csv)
- [Raw live cycles](live_cycles.json)
- [Raw static 320 profile](static_physical_320.json)
- [Route replay summary](route_replay/summary.json)
