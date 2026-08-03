"""Prepare TravelMind's Foundry Local CUDA provider and Qwen model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from foundry_runtime import (
    ensure_legacy_runtime,
    prepare_runtime,
    runtime_status,
)


_LAST_BUCKET: dict[str, int] = {}


def report_progress(name: str, percent: float) -> None:
    bucket = int(percent // 5) * 5
    if _LAST_BUCKET.get(name) == bucket and percent < 100:
        return
    _LAST_BUCKET[name] = bucket
    print(f"[FOUNDRY SETUP] {name}: {percent:.1f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TravelMind's local Foundry runtime."
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow SDK provider/model downloads when no cached legacy model exists.",
    )
    args = parser.parse_args()

    try:
        legacy = ensure_legacy_runtime(force_restart=True)
    except RuntimeError:
        if not args.allow_download:
            raise
        legacy = None
    if legacy is not None:
        endpoint, model_id = legacy
        print(f"[FOUNDRY SETUP] Reused existing model: {model_id}")
        print(f"[FOUNDRY SETUP] Endpoint: {endpoint}")
        return

    if not args.allow_download:
        raise SystemExit(
            "[FOUNDRY SETUP] No reusable legacy runtime was found. Nothing was "
            "downloaded. Re-run with --allow-download only if a new download is intended."
        )

    print("[FOUNDRY SETUP] Registering CUDA and preparing the configured model...")
    model = prepare_runtime(register_cuda=True, progress_callback=report_progress)
    print(f"[FOUNDRY SETUP] Ready: {model.id}")
    print(json.dumps(runtime_status(), indent=2))


if __name__ == "__main__":
    main()
