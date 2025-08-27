# -----------------------------
# FILE: env/utils.py
# -----------------------------

import json
import logging
import os
from typing import Any, Dict

from .errors import ConfigError

LOGGER = logging.getLogger("ysc_env")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _h.setFormatter(_fmt)
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}", hint="Check path or working directory.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}", hint="Validate JSON with a linter.")


def require(condition: bool, msg: str):
    if not condition:
        raise ConfigError(msg)
