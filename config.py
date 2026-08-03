"""Central runtime configuration for TravelMind."""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Foundry Local SDK 1.x resolves an alias to the best registered hardware
# variant.  The old qwen3-4b-cuda-gpu:2 ID belonged to the retired CLI
# catalog and is no longer a valid ID in the current SDK catalog.
MODEL_ALIAS = os.environ.get("TRAVELMIND_MODEL", "qwen3-4b")
LEGACY_MODEL_ID = os.environ.get(
    "TRAVELMIND_LEGACY_MODEL_ID", "qwen3-4b-cuda-gpu:2"
)

# Backward-compatible name used by a few tests and older imports. Runtime
# calls return the concrete loaded variant ID, never a different model alias.
MODEL_ID = MODEL_ALIAS

FOUNDRY_APP_DATA_DIR = Path(
    os.environ.get("TRAVELMIND_FOUNDRY_DATA", PROJECT_ROOT / "data" / "foundry_local")
)
_LEGACY_FOUNDRY_MODEL_CACHE = Path.home() / ".foundry" / "cache" / "models"
FOUNDRY_MODEL_CACHE_DIR = Path(
    os.environ.get(
        "TRAVELMIND_FOUNDRY_MODEL_CACHE",
        _LEGACY_FOUNDRY_MODEL_CACHE
        if _LEGACY_FOUNDRY_MODEL_CACHE.is_dir()
        else FOUNDRY_APP_DATA_DIR / "models",
    )
)
FOUNDRY_WEB_URL = os.environ.get("TRAVELMIND_FOUNDRY_URL", "http://127.0.0.1:0")
FOUNDRY_PREFER_CUDA = os.environ.get("TRAVELMIND_PREFER_CUDA", "1") != "0"
