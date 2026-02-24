"""Platform compatibility utilities for Python discovery."""

from __future__ import annotations

import functools
import logging
import os
import sys
import tempfile

IS_WIN = sys.platform == "win32"

LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def fs_is_case_sensitive() -> bool:
    with tempfile.NamedTemporaryFile(prefix="TmP") as tmp_file:
        result = not os.path.exists(tmp_file.name.lower())
    LOGGER.debug("filesystem is %scase-sensitive", "" if result else "not ")
    return result


def fs_path_id(path: str) -> str:
    return path.casefold() if not fs_is_case_sensitive() else path


__all__ = [
    "IS_WIN",
    "fs_is_case_sensitive",
    "fs_path_id",
]
