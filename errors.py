# -----------------------------
# FILE: env/errors.py
# -----------------------------

from dataclasses import dataclass

class EnvError(Exception):
    """Base class for all environment errors with a stable error code."""
    code: str = "ENV_ERROR"
    hint: str = ""

    def __init__(self, message: str = "", *, hint: str = ""):
        super().__init__(message)
        if hint:
            self.hint = hint

    def __str__(self):
        base = super().__str__()
        if self.hint:
            return f"[{self.code}] {base} | Hint: {self.hint}"
        return f"[{self.code}] {base}"

class ConfigError(EnvError):
    code = "CFG_BAD"

class MissingDependencyError(EnvError):
    code = "DEP_MISSING"

class MapLoadError(EnvError):
    code = "MAP_LOAD"

class GeneratorNotFound(EnvError):
    code = "GEN_MISSING"

class EpisodeClosedError(EnvError):
    code = "EPI_CLOSED"

class StepError(EnvError):
    code = "STEP_BAD"

@dataclass
class Suggestion:
    title: str
    details: str
