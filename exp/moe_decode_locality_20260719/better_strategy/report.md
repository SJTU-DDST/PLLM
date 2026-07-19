# Better Pause-MoE Strategy: Recent-32 Pinning

Date: 2026-07-19

All GPU tests ran on the local machine. The pre-existing `cong` route benchmark
and vLLM service were terminated before these runs.

## Strategy

Use recent-route pinning before window-LFU eviction:

1. Track the exact routed experts independently for every MoE layer.
2. Protect experts used by the most recent 32 decode token steps.
3. Rank the remaining residents with window-LFU and recency.
4. Evict only outside the current exact Top-k and the recent protected set.
5. If protection consumes every candidate, drop recent protection for that
   replacement and preserve the current exact Top-k route.

The fallback makes the optimization advisory: it can improve residency but can
never block an exact model route.

`PLLM_EER_PIN_RECENT_STEPS` configures the depth and defaults to 32. The depth
can also be changed through the local EER control socket without rebuilding
weights or touching KV/Mamba state.

## Offline route replay

The scan replays 60 captured requests containing 1,033 decode tokens and
909,040 expert accesses.

| Active slots | No pin misses/token | Pin-8 | Pin-32 |
| ---: | ---: | ---: | ---: |
| 376 | 67.59 | 65.24 | 64.63 |
| 380 | 65.16 | 62.91 | 62.32 |
| 382 | 63.92 | 61.76 | 61.18 |
| 383 | 63.32 | 61.18 | 60.61 |
| 384 | 62.74 | 60.67 | 60.09 |

At 380 slots, pin-32 has fewer misses than the old unpinned 384-slot baseline.

## Live token-boundary cycles

Each strategy used two warmups and three rotated measured rounds. All measured
requests succeeded.

| Strategy/profile | Post-transition TPOT | Same-run slowdown |
| --- | ---: | ---: |
| Old no-pin, 368 | 234.2 ms | 1.133x |
| Pin-8, 380 | 216.6 ms | 1.068x |
| Pin-32, 383 | 222.4 ms | 1.029x |
| Pin-32, 380 | 224.0 ms | 1.036x |

The absolute TPOT varied between server runs, so the strategy comparison uses
same-run normalization. Pin-32 at 380 is the selected operating point: it
reclaims more than 382/383 while remaining within about 4% of the 384 baseline.

The logical transition preserved the KV/Mamba allocation fingerprint, copied
zero state bytes, and left the runtime healthy. Logical capacity changes do not
physically release weight tensors; they measure policy behavior independently
from the known online allocator limitation.

## Static physical validation

Both profiles used the same 7.56 GiB KV/state cache.

| Physical slots | GPU memory | Mean TPOT | Relative TPOT |
| ---: | ---: | ---: | ---: |
| 384 | 66,144 MiB | 216.2 ms | 1.000x |
| 380 | 65,824 MiB | 225.3 ms | 1.042x |

The real 380-slot service reclaimed 320 MiB and was 4.2% slower. Its measured
result closely matches the logical 380 prediction.

## Decision rule

- Prefer pause-only at 384 slots when latency is the priority.
- Use recent-32 pinning with 380 slots when approximately 320 MiB of VRAM is
  useful and a roughly 4% TPOT cost is acceptable.
- Avoid larger eviction by default. The previous 368/352/320 points cost
  13%/54%/87% TPOT respectively.
- Online physical tensor reclamation remains disabled until resize executes in
  vLLM's owning allocator context. Static 380 residency is the validated
  physical configuration today.

## Artifacts

- [Normalized strategy comparison](recent_pin_strategy.png)
- [Static physical 384 versus 380](physical_moe_profiles.png)
- [Live strategy summary](recent_pin_strategy.csv)
- [Offline route scan](recent_pin_route_scan.csv)
- [Offline route scan JSON](recent_pin_route_scan.json)
- [Pin-8 live cycles](../live_cycles_pin8.json)
- [Pin-32 live cycles](../live_cycles_pin32.json)
- [Static physical 380 cycles](../static_physical_380_pin32.json)
