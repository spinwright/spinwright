import sys


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    return sys.platform == "darwin"


def require_linux(feature: str) -> None:
    if not is_linux():
        raise RuntimeError(
            f"{feature} requires Linux; current platform is {sys.platform!r}."
        )
