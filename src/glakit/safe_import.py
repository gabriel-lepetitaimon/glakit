from __future__ import annotations

import os
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2


def __getattr__(name: str) -> ModuleType:
    if name in globals():
        return globals()[name]

    match name:
        case "cv2":
            module = import_cv2()
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = module
    return module


def import_cv2(package: str | None = None) -> ModuleType:
    try:
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

        import cv2
    except ImportError:
        raise ImportError(
            "cv2 is not available in the current python environment.\n"
            f"\t It is required {'by ' + package if package else 'by the current package'} but "
            "we can't add it to the dependencies to prevent versions conflict.\n"
            "\t Please install it by yourself using `pip install opencv-python-headless`."
        )
    return cv2
