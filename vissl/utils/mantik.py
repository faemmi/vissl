import os

_TRACKING_ENV_VAR_NAME = "TRACK_TO_MANTIK"

def tracking_enabled() -> bool:
    """Return whether logging to mantik is enabled."""
    return True if os.getenv(_TRACKING_ENV_VAR_NAME) == "True" else False

def disable_tracking() -> None:
    os.environ[_TRACKING_ENV_VAR_NAME] = "False"
