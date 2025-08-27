# -----------------------------
# FILE: env/map_provider.py
# -----------------------------

import os
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .errors import MapLoadError, GeneratorNotFound, MissingDependencyError
from .utils import LOGGER

@dataclass
class MapData:
    width: int
    height: int
    resolution: float  # meters per pixel
    origin: Tuple[float, float, float]
    occ: np.ndarray            # HxW bool, True = obstacle/unknown
    free: np.ndarray           # HxW bool, True = free


def _tiny_yaml_load(path: str) -> Dict[str, Any]:
    """Very small YAML subset loader for typical map.yaml.
    Tries PyYAML first; falls back to key: value lines.
    """
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        # Fallback parser (supports only 'key: value' and simple lists like [a,b,c])
        data: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if ":" not in s:
                    continue
                k, v = s.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    # simple list
                    inside = v[1:-1].strip()
                    if inside:
                        parts = [p.strip() for p in inside.split(",")]
                        # try cast to float
                        def _cast(x):
                            try:
                                return float(x)
                            except Exception:
                                return x
                        data[k] = [_cast(p) for p in parts]
                    else:
                        data[k] = []
                else:
                    # try number cast
                    if v.lower() in ("true", "false"):
                        data[k] = (v.lower() == "true")
                    else:
                        try:
                            if "." in v or "e" in v.lower():
                                data[k] = float(v)
                            else:
                                data[k] = int(v)
                        except Exception:
                            data[k] = v
        return data


def _read_pgm(path: str) -> np.ndarray:
    """Read a binary P5 PGM to uint8 HxW. Minimal parser."""
    try:
        with open(path, "rb") as f:
            magic = f.readline().strip()
            if magic != b"P5":
                raise MapLoadError(f"Unsupported PGM magic {magic!r}", hint="Use binary P5 PGM.")
            # skip comments
            def _read_non_comment():
                line = f.readline()
                while line.startswith(b"#"):
                    line = f.readline()
                return line
            dims = _read_non_comment()
            while dims.strip() == b"":
                dims = _read_non_comment()
            w, h = [int(x) for x in dims.split()]
            maxv = int(_read_non_comment())
            if maxv > 255:
                raise MapLoadError("Only 8-bit PGM supported (maxval<=255)")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            if data.size != w*h:
                raise MapLoadError(f"PGM size mismatch: got {data.size}, expected {w*h}")
            img = data.reshape(h, w)
            return img
    except FileNotFoundError:
        raise MapLoadError(f"PGM not found: {path}", hint="Check map file path.")


def load_dataset_map(yaml_path: str, *, override_pgm: Optional[str] = None) -> MapData:
    y = _tiny_yaml_load(yaml_path)
    try:
        image_rel = y["image"]
        res = float(y["resolution"])
        origin = tuple(y.get("origin", [0.0, 0.0, 0.0]))  # x,y,theta
        negate = int(y.get("negate", 0))
        occ_thr = float(y.get("occupied_thresh", 0.65))
        free_thr = float(y.get("free_thresh", 0.196))
    except Exception as e:
        raise MapLoadError(f"Invalid map.yaml structure: {e}")

    pgm_path = override_pgm if override_pgm else os.path.join(os.path.dirname(yaml_path), image_rel)
    img = _read_pgm(pgm_path).astype(np.float32)
    # ROS map_server semantics:
    # if negate==0: p = (255 - val)/255; else p = val/255
    if negate == 0:
        p = (255.0 - img) / 255.0
    else:
        p = img / 255.0

    # Occupied if p > occ_thr; Free if p < free_thr; else unknown
    occ = p > occ_thr
    free = p < free_thr
    # Treat unknown as occupied for safety
    unknown = ~(occ | free)
    occ = occ | unknown
    free = ~occ

    h, w = free.shape
    return MapData(width=w, height=h, resolution=res, origin=origin, occ=occ, free=free)


def load_generated_map(module_name: str, kwargs: Optional[Dict[str, Any]] = None) -> MapData:
    """Call user's generator module to obtain a map.
    Expected API (flexible): module.generate(**kwargs) -> dict with keys:
      - "pgm_path" (str), "yaml_path" (str)  OR  direct arrays: "free" (HxW bool), "resolution" (float)
    If files are returned, we reuse load_dataset_map to parse thresholds consistently.
    """
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise GeneratorNotFound(
            f"Cannot import generator module '{module_name}'.",
            hint=("Put the .py next to your script or add its folder to PYTHONPATH. "
                  "Known options: random_map_generator_connected_v3, ...")
        )
    kwargs = kwargs or {}
    if hasattr(mod, "generate"):
        result = mod.generate(**kwargs)
        if isinstance(result, dict):
            if "free" in result and "resolution" in result:
                free = np.array(result["free"]).astype(bool)
                occ = ~free
                res = float(result["resolution"])
                h, w = free.shape
                return MapData(w, h, res, (0.0, 0.0, 0.0), occ, free)
            elif "yaml_path" in result:
                return load_dataset_map(result["yaml_path"], override_pgm=result.get("pgm_path"))
    # Fallback: look for emitted files in CWD
    raise MapLoadError(
        "Generator did not return expected data.",
        hint="It should return {'yaml_path': ..., 'pgm_path': ...} or {'free': array, 'resolution': mpp}."
    )
