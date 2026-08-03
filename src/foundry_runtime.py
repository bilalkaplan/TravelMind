"""Foundry Local SDK 1.x lifecycle management for TravelMind.

The SDK embeds the runtime in the Python process; there is no longer a
required ``foundry service`` daemon.  This module keeps one model and one
OpenAI-compatible web endpoint alive for the lifetime of Streamlit.
"""

from __future__ import annotations

import threading
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from openai import OpenAI

import config


_RUNTIME_LOCK = threading.RLock()
_MANAGER = None
_MODEL = None
_CLIENT = None


def find_legacy_foundry_cli() -> Path | None:
    """Locate the 0.x Foundry CLI even when its app-execution alias is gone."""
    on_path = shutil.which("foundry")
    if on_path:
        return Path(on_path)

    windows_apps = Path("C:/Program Files/WindowsApps")
    try:
        candidates = list(
            windows_apps.glob(
                "Microsoft.FoundryLocal_*_x64__8wekyb3d8bbwe/foundry.exe"
            )
        )
    except OSError:
        candidates = []
    if not candidates:
        return None

    def version_key(path: Path):
        match = re.search(r"FoundryLocal_([0-9.]+)_", str(path), re.IGNORECASE)
        if not match:
            return ()
        return tuple(int(part) for part in match.group(1).split("."))

    return max(candidates, key=version_key)


def legacy_model_is_cached(cli: Path, model_id: str) -> bool:
    """Check the legacy cache without loading or downloading a model."""
    cache_root = Path(config.FOUNDRY_MODEL_CACHE_DIR)
    model_slug = model_id.replace(":", "-").casefold()
    try:
        model_directories = [
            path
            for path in cache_root.glob("*/*")
            if path.is_dir() and path.name.casefold() == model_slug
        ]
    except OSError:
        model_directories = []
    for model_directory in model_directories:
        try:
            model_files = list(model_directory.rglob("model.onnx"))
            data_files = list(model_directory.rglob("model.onnx.data"))
            if any(path.stat().st_size > 0 for path in model_files) and any(
                path.stat().st_size > 0 for path in data_files
            ):
                return True
        except OSError:
            continue

    # Keep the CLI query as a compatibility fallback for alternate cache
    # layouts. The normal local layout returns above and avoids a slow catalog
    # refresh during Streamlit startup.
    try:
        result = subprocess.run(
            [str(cli), "cache", "list", "--log-level", "Fatal"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = f"{result.stdout}\n{result.stderr}".casefold()
    return result.returncode == 0 and model_id.casefold() in output


def ensure_legacy_runtime(*, force_restart: bool = False) -> tuple[str, str] | None:
    """Start the installed 0.x service and load the existing CUDA model.

    Returns ``(base_url, model_id)`` when the legacy CLI is installed, or
    ``None`` when callers should use the SDK 1.x fallback.
    """
    cli = find_legacy_foundry_cli()
    if cli is None:
        return None
    if not legacy_model_is_cached(cli, config.LEGACY_MODEL_ID):
        raise RuntimeError(
            f"Foundry model '{config.LEGACY_MODEL_ID}' is not present in the local "
            "cache. TravelMind will not download a model during chat startup."
        )

    if force_restart:
        subprocess.run(
            [str(cli), "service", "stop"],
            capture_output=True,
            text=True,
            check=False,
        )

    status = subprocess.run(
        [str(cli), "service", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (status.stdout or "") + "\n" + (status.stderr or "")
    endpoint_match = re.search(r"http://(?:127\.0\.0\.1|localhost):\d+", output)
    if endpoint_match is None:
        started = subprocess.run(
            [str(cli), "service", "start"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (started.stdout or "") + "\n" + (started.stderr or "")
        endpoint_match = re.search(
            r"http://(?:127\.0\.0\.1|localhost):\d+", output
        )
        if endpoint_match is None:
            status = subprocess.run(
                [str(cli), "service", "status"],
                capture_output=True,
                text=True,
                check=False,
            )
            output += "\n" + (status.stdout or "") + "\n" + (status.stderr or "")
            endpoint_match = re.search(
                r"http://(?:127\.0\.0\.1|localhost):\d+", output
            )
    if endpoint_match is None:
        raise RuntimeError(f"Foundry Local service could not start:\n{output.strip()}")

    loaded = subprocess.run(
        [str(cli), "model", "load", config.LEGACY_MODEL_ID],
        capture_output=True,
        text=True,
        check=False,
    )
    if loaded.returncode != 0:
        load_output = (loaded.stdout or "") + "\n" + (loaded.stderr or "")
        raise RuntimeError(
            f"Existing Foundry model '{config.LEGACY_MODEL_ID}' could not be loaded:\n"
            f"{load_output.strip()}"
        )

    return endpoint_match.group(0).rstrip("/") + "/v1", config.LEGACY_MODEL_ID


def _runtime_description(variant) -> tuple[str, str]:
    runtime = getattr(getattr(variant, "info", None), "runtime", None)
    device = str(getattr(runtime, "device_type", "") or "")
    provider = str(getattr(runtime, "execution_provider", "") or "")
    return device.upper(), provider.upper()


def _variant_priority(variant) -> tuple[int, int, int]:
    """Prefer a cached CUDA GPU variant, then any cached GPU variant."""
    device, provider = _runtime_description(variant)
    is_gpu = device == "GPU"
    is_cuda = "CUDA" in provider or "NVTENSORRT" in provider
    return (
        int(bool(getattr(variant, "is_cached", False))),
        int(is_gpu and is_cuda),
        int(is_gpu),
    )


def _get_manager():
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER

    from foundry_local_sdk import Configuration, FoundryLocalManager

    if FoundryLocalManager.instance is None:
        app_data = Path(config.FOUNDRY_APP_DATA_DIR)
        model_cache = Path(config.FOUNDRY_MODEL_CACHE_DIR)
        app_data.mkdir(parents=True, exist_ok=True)
        model_cache.mkdir(parents=True, exist_ok=True)
        sdk_config = Configuration(
            app_name="TravelMind",
            app_data_dir=str(app_data),
            model_cache_dir=str(model_cache),
            web=Configuration.WebService(urls=config.FOUNDRY_WEB_URL),
        )
        FoundryLocalManager.initialize(sdk_config)

    _MANAGER = FoundryLocalManager.instance
    return _MANAGER


def _resolve_model(manager, *, require_cached: bool):
    global _MODEL
    model = manager.catalog.get_model(config.MODEL_ALIAS)
    if model is None:
        raise RuntimeError(
            f"Foundry model alias '{config.MODEL_ALIAS}' is not in the local catalog. "
            "Run: .venv\\Scripts\\python.exe scripts\\setup_foundry_runtime.py"
        )

    variants = list(model.variants)
    if config.FOUNDRY_PREFER_CUDA:
        cuda_variants = [
            variant
            for variant in variants
            if _runtime_description(variant)[0] == "GPU"
            and (
                "CUDA" in _runtime_description(variant)[1]
                or "NVTENSORRT" in _runtime_description(variant)[1]
            )
        ]
        if cuda_variants:
            model.select_variant(max(cuda_variants, key=_variant_priority))
        else:
            model.select_variant(max(variants, key=_variant_priority))
    else:
        model.select_variant(max(variants, key=_variant_priority))

    if require_cached and not model.is_cached:
        raise RuntimeError(
            f"Foundry model '{model.id}' is not downloaded. Run: "
            ".venv\\Scripts\\python.exe scripts\\setup_foundry_runtime.py"
        )

    _MODEL = model
    return model


def ensure_runtime(*, force_restart: bool = False):
    """Return ``(OpenAI client, concrete model ID, base URL)``.

    The model must already have been prepared by ``setup_foundry_runtime.py``.
    No network download is ever triggered from a user's chat request.
    """
    global _CLIENT
    with _RUNTIME_LOCK:
        manager = _get_manager()
        model = _resolve_model(manager, require_cached=True)

        if force_restart:
            if manager.urls:
                manager.stop_web_service()
            for loaded_model in manager.catalog.get_loaded_models():
                loaded_model.unload()
            _CLIENT = None

        if not model.is_loaded:
            model.load()
        if not manager.urls:
            manager.start_web_service()

        base_url = manager.urls[0].rstrip("/") + "/v1"
        if _CLIENT is None:
            _CLIENT = OpenAI(base_url=base_url, api_key="not-needed", timeout=300.0)
        return _CLIENT, model.id, base_url


def prepare_runtime(
    *,
    register_cuda: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
):
    """Register CUDA when available and download the selected model variant."""
    global _MODEL
    with _RUNTIME_LOCK:
        manager = _get_manager()

        if register_cuda:
            eps = manager.discover_eps()
            cuda = next((ep for ep in eps if ep.name == "CUDAExecutionProvider"), None)
            if cuda is not None and not cuda.is_registered:
                callback = None
                if progress_callback is not None:
                    callback = lambda name, percent: progress_callback(name, percent)
                manager.download_and_register_eps([cuda.name], progress_callback=callback)

        # EP registration invalidates the SDK catalog; resolve again so GPU
        # variants become visible before choosing what to download.
        _MODEL = None
        model = _resolve_model(manager, require_cached=False)
        if not model.is_cached:
            callback = None
            if progress_callback is not None:
                callback = lambda percent: progress_callback(model.id, percent)
            model.download(callback)
        return model


def runtime_status() -> dict:
    """Return a serializable snapshot without loading a model."""
    with _RUNTIME_LOCK:
        manager = _get_manager()
        eps = manager.discover_eps()
        model = _resolve_model(manager, require_cached=False)
        device, provider = _runtime_description(model)
        return {
            "model_alias": config.MODEL_ALIAS,
            "model_id": model.id,
            "cached": bool(model.is_cached),
            "loaded": bool(model.is_loaded),
            "device": device,
            "execution_provider": provider,
            "service_urls": list(manager.urls or []),
            "execution_providers": {
                ep.name: bool(ep.is_registered) for ep in eps
            },
        }
