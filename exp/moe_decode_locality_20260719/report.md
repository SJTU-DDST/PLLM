# MoE decode locality experiment

## Evidence

- Route source: `live_full_resident_routes_with_offline_exact_cache_replay`
- Successful requests: 60
- Route files: 60
- Captured decode tokens: 1033
- Cache replay preserves every actual Top-k route; a miss is counted as a blocking load.

## Result

The most aggressive tested profile that met both guardrails was window_lfu at 448 slots/layer: 96.83% byte hit, 7.38 GiB projected reclaim, 27.93 blocking loads/token, and 3.09x estimated TPOT.

With 128 prior tokens, 95.48% of current expert accesses had appeared in the same layer's history, while that history contained 401.3 distinct experts/layer on average.

## Scope

The hit and reclaim results are offline replay of live full-resident routes. Latency is an estimate from configured I/O bandwidth and p95 per-object latency; it is not a live elastic-throughput measurement.
