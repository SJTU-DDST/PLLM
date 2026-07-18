# Results

- `mock_integration.json` records the GPU-safe `mode=keep` control validation.
- `rdma_bench.json` separates registered-host staging from an unrun network test.
- `pllm-dashboard-v2.png` and `pllm-dashboard-mobile-v2.png` are real Vue renders.
- `pllm-dashboard-live-desktop.png` and `pllm-dashboard-live-mobile.png` render
  the real Nemotron/EER daemon rather than mock data.
- `pllm-overlay-v2.png` is the updated PySide6 overlay render.
- `ui-demo.png` is retained from the first prototype for comparison.
- `nemotron_eer128_summary.json` summarizes the real 120B NVFP4 + 128-slot EER run.
- `nemotron_eer128_level1.json` and `nemotron_eer128_level2.json` contain real sleep measurements.
- `nemotron_eer128_continuity_level0.json` and
  `nemotron_eer128_determinism_control.json` preserve both the same-stream result
  and the independent-request determinism limitation.
- `rdma_store_live.json` records real RC RDMA PUT/GET integrity and keeps host
  staging copy bandwidth separate from unmeasured network bandwidth.
- `nemotron_foreground_admission.json` records the real 60GiB CUDA allocation
  changing from OOM while resident to success after PLLM Level 2.
- `nemotron_calibration.json` is the Level 2 cost profile derived from the live
  segmented sleep/reload/wake measurement.
- `rdma_memory_pool_live_put_sharded.json` and
  `rdma_memory_pool_live_get_sharded.json` record the 75-to-71 four-QP volatile
  pool run over 15,858,978,307 bytes of real runtime-expert objects. Their
  primary metrics are process-level wall time: 2.545s PUT and 4.012s GET.
- The cross-host pool files are protocol-v1 evidence. Local v2 smoke results
  must not be used to claim a cross-host speedup until the same profile is rerun.
- `qa_benchmark/native_full/` contains all 150 per-sample MQA/NQA/TQA
  predictions and the aggregate full-resident baseline. `qa_benchmark/pllm_eer128/`
  and `qa_benchmark/pllm_eer256/` are censored release-capable trials, not zero-F1
  completed runs. `qa_benchmark/pllm_full_sleep_startup_failure.json` records the
  full-resident Sleep Mode initialization OOM.
- `qa_benchmark/experiment_matrix.json` is the compact machine-readable index.
  It uses JSON `null` for quality metrics from startup failures and censored runs.
