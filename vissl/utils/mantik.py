import os

_TRACKING_ENV_VAR = "TRACK_TO_MANTIK"
_CURRENT_EPOCH_ENV_VAR = "CURRENT_EPOCH"


def tracking_enabled() -> bool:
    """Return whether logging to mantik is enabled."""
    return True if os.getenv(_TRACKING_ENV_VAR) == "True" else False


def disable_tracking() -> None:
    os.environ[_TRACKING_ENV_VAR] = "False"


def set_current_epoch(epoch: int) -> None:
    os.environ[_CURRENT_EPOCH_ENV_VAR] = str(epoch)


def get_current_epoch() -> int:
    return int(_get_required_env_var(_CURRENT_EPOCH_ENV_VAR))


def _get_required_env_var(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Environment variable {name} unset")
    return int(value)
