from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path


VERSION = "0.25.1"
MARKER = "PLLM_HIBERCACHE_PRESERVE_CONNECTOR"
ORIGINAL = """        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
"""
PATCHED = """        clear_prefix_cache = level >= 1
        # PLLM_HIBERCACHE_PRESERVE_CONNECTOR: drain transient jobs and reset the
        # bounded CPU primary while durable active blocks remain in secondary
        # tiers. Keeping primary metadata without a connector quiesce is unsafe.
        preserve_connector = level >= 1 and mode == "keep"
        pause_future = self.pause_scheduler(
            mode=mode,
            clear_cache=clear_prefix_cache and not preserve_connector,
        )
        if preserve_connector:
            self._reset_caches(
                reset_running_requests=True,
                reset_connector=True,
            )
"""
LEGACY_PATCHES = (
    """        clear_prefix_cache = level >= 1
        # PLLM_HIBERCACHE_PRESERVE_CONNECTOR: retain the bounded CPU hot tier
        # and durable secondary tiers across deep mode=keep. Set the environment
        # switch to 0 for the cold-tier baseline or under host-memory pressure.
        keep_cpu_tier = os.getenv("PLLM_HIBERCACHE_PRESERVE_CPU_TIER", "1") not in {
            "0",
            "false",
            "no",
        }
        preserve_connector = level >= 1 and mode == "keep" and keep_cpu_tier
        pause_future = self.pause_scheduler(
            mode=mode,
            clear_cache=clear_prefix_cache and not preserve_connector,
        )
        if preserve_connector:
            self._reset_caches(
                reset_running_requests=True,
                reset_connector=False,
            )
""",
    """        clear_prefix_cache = level >= 1
        # PLLM_HIBERCACHE_PRESERVE_CONNECTOR: keep durable KV tiers across
        # a deep mode=keep sleep while resetting only resident cache state.
        preserve_connector = level >= 1 and mode == "keep"
        pause_future = self.pause_scheduler(
            mode=mode,
            clear_cache=clear_prefix_cache and not preserve_connector,
        )
        if preserve_connector:
            self._reset_caches(
                reset_running_requests=True,
                reset_connector=False,
            )
""",
    """        clear_prefix_cache = level >= 1
        # PLLM_HIBERCACHE_PRESERVE_CONNECTOR: reset transient jobs and the
        # primary tier while TieringOffloadingManager keeps durable FS/network
        # secondary tiers across a deep mode=keep sleep.
        preserve_connector = level >= 1 and mode == "keep"
        pause_future = self.pause_scheduler(
            mode=mode,
            clear_cache=clear_prefix_cache and not preserve_connector,
        )
        if preserve_connector:
            self._reset_caches(
                reset_running_requests=True,
                reset_connector=True,
            )
""",
)


def target_path() -> Path:
    distribution = importlib.metadata.distribution("vllm")
    version = distribution.version
    if version != VERSION:
        raise RuntimeError(f"Expected vLLM {VERSION}, found {version}")
    return Path(distribution.locate_file("vllm/v1/engine/core.py"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the PLLM HiberCache vLLM patch")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    path = target_path()
    text = path.read_text(encoding="utf-8")
    installed = PATCHED in text
    legacy = next((block for block in LEGACY_PATCHES if block in text), None)
    if args.check:
        state = "installed" if installed else "legacy" if legacy else "missing"
        print(f"hibercache_patch={state} target={path}")
        raise SystemExit(0 if installed else 1)

    if args.revert:
        if not installed and legacy is None:
            print(f"HiberCache patch is not installed: {path}")
            return
        block = PATCHED if installed else legacy
        assert block is not None
        path.write_text(text.replace(block, ORIGINAL, 1), encoding="utf-8")
        print(f"Reverted HiberCache patch: {path}")
        return

    if installed:
        print(f"HiberCache patch already installed: {path}")
        return
    if legacy is not None:
        path.write_text(text.replace(legacy, PATCHED, 1), encoding="utf-8")
        print(f"Updated HiberCache patch: {path}")
        return
    if ORIGINAL not in text:
        raise RuntimeError("vLLM sleep implementation does not match the guarded patch")
    backup = path.with_suffix(path.suffix + ".pllm.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    print(f"Installed HiberCache patch: {path}")


if __name__ == "__main__":
    main()
