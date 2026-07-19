const { createApp } = Vue;

const EMPTY_STATUS = {
  state: "active",
  mode: "auto",
  reason: "",
  workload: "idle",
  pause_mode: "keep",
  sleep_level: 0,
  transition_progress: 0,
  reclaimed_gb: null,
  sensor: {},
  decision: {},
  hibercache: {},
  expert_residency: { available: false, data_plane_ready: false, plan: {}, model: {}, data_plane: {} },
  services: [],
};

const DEMO_STEPS = [
  { state: "active", workload: "idle", reason: "Prefill 全驻留：Nemotron 正在生成 CUDA 代码", gpu: 68, mem: 28, progress: 0, token: 184, phase: "prefill", observations: 0, expert: { action: "full_resident", slots: 512, resident: 74.783, reclaim: 0, hit: 1, slowdown: 1 } },
  { state: "yielding", workload: "creative", reason: "Blender 激活；prefill 护栏先在 token 边界冻结，不驱逐专家", gpu: 52, mem: 49, progress: 22, token: 193, phase: "prefill", observations: 0, expert: { action: "full_resident", slots: 512, resident: 74.783, reclaim: 0, hit: 1, slowdown: 1 } },
  { state: "active", workload: "creative", reason: "进入 decode，采集 320 个 layer-step 路由观测", gpu: 48, mem: 46, progress: 38, token: 201, phase: "decode", observations: 320, expert: { action: "observe", slots: 512, resident: 74.783, reclaim: 0, hit: 1, slowdown: 1 } },
  { state: "elastic_resident", workload: "creative", reason: "候选轨迹：496 slots 通过命中率与延迟护栏，状态小岛保持原位", gpu: 41, mem: 51, progress: 72, token: 215, phase: "decode", observations: 880, expert: { action: "decode_elastic", slots: 496, resident: 72.937, reclaim: 1.846, hit: 0.988, slowdown: 1.61 } },
  { state: "hibernated", workload: "memory_pressure", reason: "若 1.85 GiB 不足且所有档位超预算，则拒绝抖动并升级 Level 2", gpu: 8, mem: 18, progress: 100, token: 215, phase: "idle", observations: 880, reclaimed: 72.0, expert: { action: "yield", slots: 512, resident: 74.783, reclaim: 0, hit: 0.988, slowdown: null } },
  { state: "restoring", workload: "idle", reason: "前台退出；优先从 Remote DRAM 暖源恢复，NVMe 仅回退", gpu: 34, mem: 24, progress: 63, token: 215, phase: "idle", observations: 0, reclaimed: 72.0, expert: { action: "full_resident", slots: 512, resident: 74.783, reclaim: 0, hit: 1, slowdown: 1 } },
  { state: "active", workload: "idle", reason: "恢复完整驻留，请求从 token 215 继续", gpu: 61, mem: 27, progress: 100, token: 239, phase: "decode", observations: 320, reclaimed: 0, expert: { action: "full_resident", slots: 512, resident: 74.783, reclaim: 0, hit: 1, slowdown: 1 } },
];

createApp({
  data() {
    return {
      status: structuredClone(EMPTY_STATUS),
      replayStatus: structuredClone(EMPTY_STATUS),
      capabilities: {},
      events: [],
      experiments: [],
      replays: [],
      samples: [],
      judgeMode: false,
      demoPlaying: false,
      demoIndex: 0,
      expertPreset: "creative",
      expertActionBusy: false,
      busy: false,
      toast: null,
      policyText: "Blender 渲染时优先释放全部模型资源",
      compiledPolicy: "",
      eventSource: null,
      pollTimer: null,
      demoTimer: null,
    };
  },
  computed: {
    activeStatus() { return this.judgeMode ? this.replayStatus : this.status; },
    sensor() { return this.activeStatus.sensor || {}; },
    stateClass() { return `state-${this.activeStatus.state || "error"}`; },
    backendAvailable() {
      return this.judgeMode || (this.activeStatus.services || []).some((service) => service.healthy && service.controllable);
    },
    stateLabel() {
      if (this.activeStatus.state === "active" && !this.backendAvailable) return "监控运行中";
      return ({ active: "推理运行中", elastic_resident: "弹性专家驻留", yielding: "Token 边界冻结", quiescing: "正在释放资源", hibernated: "已深度休眠", restoring: "正在恢复", error: "需要处理" })[this.activeStatus.state] || this.activeStatus.state;
    },
    mockBackend() {
      return !this.judgeMode && (this.activeStatus.services || []).some((service) => String(service.model || "").startsWith("mock-"));
    },
    dataModeLabel() {
      if (this.judgeMode) return "SCENARIO REPLAY · NOT MEASURED";
      return this.mockBackend ? "LIVE SENSORS · MOCK BACKEND" : "LIVE CONTROL";
    },
    headline() {
      if (this.mockBackend && this.activeStatus.state === "hibernated") return "Mock 控制路径已休眠";
      if (this.activeStatus.state === "active" && !this.backendAvailable) return "等待可控 vLLM 服务";
      return ({ active: "后台 AI 随时可用", elastic_resident: "专家工作集正在适配前台任务", yielding: "请求已冻结，输出流保持连接", quiescing: "正在将资源让给前台任务", hibernated: "模型资源已释放", restoring: "正在恢复模型与请求状态", error: "控制路径发生异常" })[this.activeStatus.state] || "PLLM 正在监控系统";
    },
    foregroundName() {
      const foreground = this.sensor.foreground || {};
      if (this.judgeMode) return this.demoIndex >= 1 && this.demoIndex <= 4 ? "Blender" : "GNOME Desktop";
      return foreground.wm_class || foreground.app_id || foreground.title || "未识别";
    },
    memoryMetric() {
      const value = this.sensor.uma ? this.sensor.memory_available_gb : this.sensor.gpu_memory_used_gb;
      return this.metric(value, " GiB", 1);
    },
    durationText() {
      const value = this.activeStatus.last_action_duration_ms;
      return value == null ? "等待释放动作" : `${Number(value).toFixed(0)} ms action`;
    },
    transitionProgress() {
      const raw = Number(this.activeStatus.transition_progress || 0);
      return Math.max(0, Math.min(100, raw <= 1 ? raw * 100 : raw));
    },
    expertResidency() { return this.activeStatus.expert_residency || {}; },
    capacityPlan() { return this.expertResidency.plan || {}; },
    decodePlan() { return this.expertResidency.decode_plan || {}; },
    expertPlan() {
      return this.decodePlan.action ? this.decodePlan : this.capacityPlan;
    },
    expertModel() { return this.expertResidency.model || {}; },
    expertDataPlane() { return this.expertResidency.data_plane || {}; },
    routeTrace() { return this.expertDataPlane.route_trace || {}; },
    stateIsland() { return this.expertDataPlane.state_island || {}; },
    expertActiveSlots() {
      if (this.expertResidency.data_plane_ready) {
        return Number(this.expertDataPlane.slots_per_layer ?? 0);
      }
      return Number(this.expertPlan.slots_per_layer ?? 0);
    },
    expertEvidenceLabel() {
      if (this.expertResidency.data_plane_ready) return "LIVE DATA PLANE";
      if (this.judgeMode) return "SCENARIO · NOT MEASURED";
      return "CONTROL PLANE ONLY";
    },
    expertLayerCells() {
      const count = Number(this.expertModel.moe_layers || 40);
      const total = Number(this.expertModel.experts_per_layer || 512);
      const slots = this.expertActiveSlots || total;
      const fill = Math.max(0, Math.min(100, slots / Math.max(1, total) * 100));
      return Array.from({ length: count }, (_, index) => ({ index: index + 1, fill, slots }));
    },
    tiers() {
      const caps = this.capabilities;
      const hiber = this.activeStatus.hibercache || {};
      const rdma = caps.rdma || {};
      const devices = rdma.devices || [];
      const selected = this.activeStatus.restore_source || "remote_dram";
      const islandBytes = Number(this.stateIsland.allocated_bytes || 0);
      const sourceActive = ["quiescing", "hibernated", "restoring"].includes(this.activeStatus.state);
      return [
        { role: "ACTIVE", name: this.sensor.uma ? "Local UMA" : "GPU VRAM", icon: "cpu", detail: "非专家权重 + decode 热专家", value: this.sensor.uma ? "coherent" : this.metric(this.sensor.gpu_memory_used_gb, " GiB", 1), selected: ["active", "elastic_resident"].includes(this.activeStatus.state), available: !!this.sensor.gpu_available, linkActive: ["elastic_resident", "yielding", "quiescing", "restoring"].includes(this.activeStatus.state) },
        { role: "LIVE STATE", name: "KV / Mamba island", icon: "memory-stick", detail: "专家 resize 前后 allocation 不变", value: islandBytes ? this.metric(islandBytes / 1024 / 1024, " MiB", 0) : "guarded", selected: ["active", "elastic_resident", "yielding"].includes(this.activeStatus.state), available: this.stateIsland.attached !== false, linkActive: ["quiescing", "restoring"].includes(this.activeStatus.state) },
        { role: "WARM SOURCE", name: "Remote DRAM", icon: "network", detail: devices[0] ? `${devices[0].name} · ${devices[0].rate} · host staged` : "ConnectX host-staged RDMA", value: devices.length ? "ready" : "fallback", selected: sourceActive && selected === "remote_dram", available: devices.length > 0, linkActive: sourceActive && selected === "local_nvme" },
        { role: "FALLBACK", name: "Local NVMe", icon: "hard-drive", detail: hiber.root || "/mnt/ssd-storage/pllm-cache", value: this.metric(hiber.used_gb, " GiB", 2), selected: sourceActive && selected === "local_nvme", available: hiber.enabled !== false, linkActive: false },
      ];
    },
    costRows() {
      const costs = this.activeStatus.decision?.costs || { yield: 0.42, hibernate: 0.71, restore_penalty: 0.12 };
      const rows = [["yield", "微暂停"], ["hibernate", "深度休眠"], ["restore_penalty", "恢复代价"]];
      const max = Math.max(1, ...rows.map(([key]) => Number(costs[key] || 0)));
      return rows.map(([name, label]) => ({ name, label, value: Number(costs[name] || 0).toFixed(3), width: Math.max(4, Number(costs[name] || 0) / max * 100) }));
    },
    decisionScore() { const value = this.activeStatus.decision?.score; return value == null ? "--" : Number(value).toFixed(3); },
    decisionReason() { return this.activeStatus.decision?.reason || this.activeStatus.reason || "等待下一次资源竞争信号"; },
    lifecycle() {
      const ids = ["active", "elastic_resident", "yielding", "quiescing", "hibernated", "restoring"];
      const labels = ["ACTIVE", "ELASTIC", "YIELD", "COMMIT", "HIBERNATED", "RESTORE"];
      const details = ["完整驻留", "专家收缩", "冻结", "事务提交", "慢层驻留", "渐进装载"];
      const current = ids.indexOf(this.activeStatus.state);
      return ids.map((id, index) => ({ id, label: labels[index], detail: details[index], active: id === this.activeStatus.state, done: current > index || (this.activeStatus.state === "active" && this.demoIndex > 0) }));
    },
    activeReplays() {
      if (this.judgeMode) return [{ id: "demo-request", status: this.activeStatus.state === "active" ? "running" : "paused", request: { messages: [{ content: "实现一个 CUDA kernel 并解释性能瓶颈" }] }, generated_tokens: DEMO_STEPS[this.demoIndex]?.token, paused_at_token: DEMO_STEPS[this.demoIndex]?.token }];
      return this.replays.filter((item) => ["running", "paused", "queued", "aborted"].includes(item.status));
    },
    capabilityRows() {
      const caps = this.capabilities;
      const loader = caps.sparkload || {};
      const patch = caps.vllm?.hibercache_patch || {};
      const rdma = caps.rdma || {};
      const stateIsland = caps.hibercache?.active_state_island || {};
      return [
        { name: "vLLM Sleep Mode", ready: caps.vllm?.version === "0.25.1", detail: caps.vllm?.version || "not detected" },
        { name: "HiberCache patch", ready: !!patch.installed, detail: patch.installed ? "guarded · GPU validation pending" : "token recompute fallback" },
        { name: "SparkLoad", ready: !!loader.fastsafetensors_version, detail: loader.selected || "multithread fallback" },
        { name: "Host-staged RDMA", ready: (rdma.devices || []).length > 0, detail: caps.cuda?.rdma_path === "host_staging" ? "host staging enforced" : `${(rdma.devices || []).length} device(s)` },
        { name: "KV/Mamba state island", ready: stateIsland.weight_independent === true, detail: stateIsland.deep_sleep_exact_resume_validated ? "exact deep resume validated" : "decode resize isolated · deep resume pending" },
        { name: "Expert slot data plane", ready: !!this.expertResidency.data_plane_ready, detail: this.expertResidency.data_plane_ready ? "physical slots active" : "control-plane projection" },
        { name: "Nemotron continuity", ready: !!caps.continuity?.real_model_validated, detail: caps.continuity?.real_model_validated ? "greedy exact match" : "pending GPU experiment" },
      ];
    },
    capabilityReady() { return this.capabilityRows.filter((item) => item.ready).length; },
    displayExperiments() {
      if (!this.judgeMode) return this.experiments.slice(0, 5);
      return [
        { id: "demo-1", name: "Decode 496-slot policy", variant: "scenario · not measured", created_at: Date.now() / 1000, metrics: { projected_reclaim_gib: 1.846 } },
        { id: "demo-2", name: "stream continuity", variant: "mock validated", created_at: Date.now() / 1000 - 80, metrics: { duplicate_tokens: 0, missing_tokens: 0 } },
        ...this.experiments.slice(0, 3),
      ];
    },
    displayEvents() {
      if (!this.judgeMode) return this.events;
      return [
        { id: "d1", event_type: "expert_dataplane", reason: "scenario: decode 496-slot candidate", created_at: Date.now() / 1000 - 22 },
        { id: "d2", event_type: "wake", reason: "scenario: Remote DRAM warm restore", created_at: Date.now() / 1000 - 11 },
        ...this.events,
      ];
    },
    eventMarker() { return Math.max(0, Math.min(720, this.demoIndex * 144)); },
  },
  methods: {
    metric(value, suffix = "", precision = 0) { return value == null || Number.isNaN(Number(value)) ? "--" : `${Number(value).toFixed(precision)}${suffix}`; },
    percent(value) { return value == null || Number.isNaN(Number(value)) ? "--" : `${(Number(value) * 100).toFixed(2)}%`; },
    ratio(value) { return value == null || !Number.isFinite(Number(value)) ? "fallback" : `${Number(value).toFixed(2)}x`; },
    formatBytes(value) {
      const bytes = Number(value || 0);
      if (!bytes) return "--";
      return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
    },
    async api(path, options = {}) {
      const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload;
    },
    async loadSupportingData() {
      try {
        const [events, replays, experiments] = await Promise.all([
          this.api("/api/v1/events?limit=30"), this.api("/api/v1/replays?limit=30"), this.api("/api/v1/experiments?limit=20"),
        ]);
        this.events = events.events || [];
        this.replays = replays.replays || [];
        this.experiments = experiments.experiments || [];
      } catch (error) { this.showToast(error.message, "error"); }
    },
    async loadCapabilities(refresh = false) {
      try { this.capabilities = await this.api(`/api/v1/capabilities${refresh ? "?refresh=1" : ""}`); this.refreshIcons(); }
      catch (error) { this.showToast(error.message, "error"); }
    },
    connectTelemetry() {
      if (this.eventSource) this.eventSource.close();
      this.eventSource = new EventSource("/api/v1/telemetry/stream?interval=0.5");
      this.eventSource.addEventListener("status", (event) => { this.status = JSON.parse(event.data); this.pushSample(this.status); this.refreshIcons(); });
      this.eventSource.onerror = () => {
        this.eventSource?.close(); this.eventSource = null;
        if (!this.pollTimer) this.pollTimer = window.setInterval(() => this.pollStatus(), 1500);
      };
    },
    async pollStatus() {
      try { this.status = await this.api("/api/v1/status"); this.pushSample(this.status); }
      catch (_) { /* visible through stale live badge */ }
    },
    pushSample(status) {
      const sensor = status.sensor || {};
      const available = Number(sensor.memory_available_gb || 0);
      const total = Math.max(1, Number(sensor.memory_total_gb || 1));
      this.samples.push({ gpu: Number(sensor.gpu_util || 0), memory: Math.max(0, Math.min(100, 100 - available / total * 100)) });
      if (this.samples.length > 90) this.samples.shift();
    },
    linePoints(key) {
      const rows = this.samples.length > 1 ? this.samples : [{ gpu: 0, memory: 0 }, { gpu: 0, memory: 0 }];
      return rows.map((row, index) => `${index / Math.max(1, rows.length - 1) * 720},${210 - Number(row[key] || 0) * 1.9}`).join(" ");
    },
    async runAction(action) {
      if (this.judgeMode && action !== "benchmark") { this.showToast("评委模式不会修改实时服务", "info"); return; }
      this.busy = true;
      try {
        const result = await this.api("/api/v1/actions", { method: "POST", body: JSON.stringify({ action }) });
        if (action === "benchmark" && result.experiment_id) await this.loadSupportingData(); else this.status = result;
        this.showToast(action === "benchmark" ? "无 GPU 基准已记录" : "控制动作已提交", "success");
      } catch (error) { this.showToast(error.message, "error"); }
      finally { this.busy = false; this.refreshIcons(); }
    },
    async planExpertPreset(preset) {
      this.expertPreset = preset;
      const inputs = {
        idle: { workload: "idle", byte_hit_rate: 1, false_prefetch_ratio: 0, envelope: { foreground_reserve_gib: 20, compute_duty_cycle: 1, io_budget_gib_s: 2 } },
        creative: { workload: "creative", byte_hit_rate: 0.95, false_prefetch_ratio: 0.05, envelope: { foreground_reserve_gib: 64, compute_duty_cycle: 0.35, io_budget_gib_s: 2 } },
        emergency: { workload: "memory_pressure", byte_hit_rate: 0.8, false_prefetch_ratio: 0.2, envelope: { foreground_reserve_gib: 104, compute_duty_cycle: 0.1, io_budget_gib_s: 0.5 } },
      };
      if (this.judgeMode) {
        const demoMap = { idle: 0, creative: 1, emergency: 3 };
        this.demoIndex = demoMap[preset]; this.applyDemoStep(this.demoIndex); return;
      }
      try {
        const result = await this.api("/api/v1/expert-residency/plan", { method: "POST", body: JSON.stringify(inputs[preset]) });
        this.status.expert_residency = result;
        this.showToast(
          result.plan?.executable
            ? "已生成可手动应用的物理槽位计划"
            : "已生成不可执行的控制面投影",
          "success",
        );
      } catch (error) { this.showToast(error.message, "error"); }
      this.refreshIcons();
    },
    async applyExpertPlan() {
      const slots = Number(this.expertPlan.slots_per_layer || 0);
      if (!this.expertPlan.executable || slots < 22) return;
      this.expertActionBusy = true;
      try {
        await this.api("/api/v1/expert-dataplane/actions", {
          method: "POST",
          body: JSON.stringify({ action: "resize", slots_per_layer: slots, retain_policy: this.expertPlan.action === "decode_elastic" ? "decode_hot" : "lru" }),
        });
        await Promise.all([this.pollStatus(), this.loadCapabilities(true)]);
        this.showToast("物理专家槽位已切换", "success");
      } catch (error) {
        this.showToast(error.message, "error");
      } finally {
        this.expertActionBusy = false;
        this.refreshIcons();
      }
    },
    setJudgeMode(enabled) {
      this.judgeMode = enabled;
      this.stopJudgeDemo();
      if (enabled) { this.demoIndex = 0; this.applyDemoStep(0); }
      this.refreshIcons();
    },
    playJudgeDemo() {
      this.stopJudgeDemo(); this.demoPlaying = true; this.demoIndex = 0; this.samples = [];
      this.applyDemoStep(0);
      this.demoTimer = window.setInterval(() => {
        this.demoIndex += 1;
        if (this.demoIndex >= DEMO_STEPS.length) { this.stopJudgeDemo(); this.demoIndex = DEMO_STEPS.length - 1; return; }
        this.applyDemoStep(this.demoIndex);
      }, 1300);
    },
    stopJudgeDemo() { if (this.demoTimer) window.clearInterval(this.demoTimer); this.demoTimer = null; this.demoPlaying = false; },
    applyDemoStep(index) {
      const step = DEMO_STEPS[index];
      this.replayStatus = { ...structuredClone(EMPTY_STATUS), state: step.state, workload: step.workload, reason: step.reason, transition_progress: step.progress, reclaimed_gb: step.reclaimed ?? null, last_action_duration_ms: null, restore_source: "remote_dram", pause_mode: "keep", sleep_level: ["quiescing", "hibernated", "restoring"].includes(step.state) ? 2 : 0, sensor: { gpu_available: true, gpu_name: "NVIDIA GB10", gpu_util: step.gpu, memory_total_gb: 128, memory_available_gb: 128 - step.mem, power_watts: step.state === "hibernated" ? 42 : 87, uma: true, foreground: { wm_class: index >= 1 && index <= 4 ? "Blender" : "GNOME Desktop" } }, hibercache: { enabled: true, root: "/mnt/ssd-storage/pllm-cache", used_gb: 12.8 }, expert_residency: { available: true, backend: "scenario_replay", data_plane_ready: false, evidence: "policy_scenario_not_measurement", model: { moe_layers: 40, experts_per_layer: 512, top_k: 22, routed_expert_gib: 59.063, non_routed_gib: 15.72, average_expert_mib: 2.953 }, plan: { action: "elastic_resident", slots_per_layer: step.expert.slots, evidence: "policy_scenario_not_measurement" }, decode_plan: { action: step.expert.action, slots_per_layer: step.expert.slots, resident_weight_gib: step.expert.resident, projected_reclaim_gib: step.expert.reclaim, projected_byte_hit_rate: step.expert.hit, estimated_slowdown_ratio: step.expert.slowdown, observations: step.observations, exact_route_required: true, executable: false, data_plane_ready: false, evidence: "policy_scenario_not_measurement" }, data_plane: { route_trace: { phase: step.phase, decode_observations: step.observations, projected_byte_hit_rate: { [step.expert.slots]: step.expert.hit } }, state_island: { attached: true, allocated_bytes: 441450496, copy_bytes: 0, resize_guard: { checked: true, preserved: true } } } }, decision: { score: step.state === "yielding" ? 0.42 : 0.71, reason: step.reason, costs: { yield: step.state === "yielding" ? 0.42 : 2.84, hibernate: 0.71, restore_penalty: 0.12 } } };
      this.pushSample(this.replayStatus); this.refreshIcons();
    },
    async compilePolicy() {
      this.busy = true;
      try {
        const result = await this.api("/api/v1/policy/compile", { method: "POST", body: JSON.stringify({ text: this.policyText, apply: false }) });
        this.compiledPolicy = `${result.rules.map((rule) => `${rule.workload}: ${rule.action}`).join(" · ")} · ${result.safety}`;
        this.showToast("策略已通过本地护栏编译", "success");
      } catch (error) { this.showToast(error.message, "error"); }
      finally { this.busy = false; }
    },
    requestTitle(item) {
      const messages = item.request?.messages || [];
      const text = messages[messages.length - 1]?.content || item.id;
      return String(text).slice(0, 48);
    },
    replayState(item) { return item.status === "paused" ? "流保持连接，等待恢复" : item.status; },
    experimentResult(item) {
      const metrics = typeof item.metrics === "string" ? JSON.parse(item.metrics || "{}") : (item.metrics || item.payload || {});
      if (metrics.latency_ms != null) return `${Number(metrics.latency_ms).toFixed(0)} ms`;
      if (metrics.rdma_host_staging?.staging_bandwidth_gbps != null) return `${Number(metrics.rdma_host_staging.staging_bandwidth_gbps).toFixed(1)} Gb/s`;
      if (metrics.storage?.bandwidth_gbps != null) return `${Number(metrics.storage.bandwidth_gbps).toFixed(2)} Gb/s`;
      if (metrics.read_gbps != null) return `${Number(metrics.read_gbps).toFixed(2)} GB/s`;
      if (metrics.projected_reclaim_gib != null) return `${Number(metrics.projected_reclaim_gib).toFixed(2)} GiB projected`;
      if (metrics.duplicate_tokens === 0) return "0 token loss";
      return "recorded";
    },
    shortTime(value) {
      if (!value) return "--";
      const date = new Date(Number(value) > 1e12 ? Number(value) : Number(value) * 1000);
      return Number.isNaN(date.valueOf()) ? String(value).slice(0, 16) : date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    },
    eventName(type) { return ({ sleep: "休眠", hibernate: "深度休眠", wake: "恢复", benchmark: "基准", policy: "策略", expert_dataplane: "专家驻留" })[type] || type || "事件"; },
    eventTone(type) { return ["wake", "benchmark"].includes(type) ? "good" : type?.includes("error") ? "bad" : "warn"; },
    showToast(text, tone = "info") { this.toast = { text, tone }; window.clearTimeout(this.toastTimer); this.toastTimer = window.setTimeout(() => { this.toast = null; }, 3200); },
    refreshIcons() { this.$nextTick(() => window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } })); },
  },
  async mounted() {
    await Promise.all([this.pollStatus(), this.loadCapabilities(), this.loadSupportingData()]);
    this.connectTelemetry(); this.refreshIcons();
    window.setInterval(() => this.loadSupportingData(), 5000);
  },
  beforeUnmount() { this.eventSource?.close(); window.clearInterval(this.pollTimer); this.stopJudgeDemo(); },
}).mount("#app");
