"""Cache Protocol and built-in implementations for Python interpreter discovery."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager, suppress
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class ContentStore(Protocol):
    """A store for reading and writing cached content."""

    def exists(self) -> bool: ...

    def read(self) -> dict | None: ...

    def write(self, content: dict) -> None: ...

    def remove(self) -> None: ...

    @contextmanager
    def locked(self) -> Generator[None]: ...


@runtime_checkable
class PyInfoCache(Protocol):
    """Cache interface for Python interpreter information."""

    def py_info(self, path: Path) -> ContentStore: ...

    def py_info_clear(self) -> None: ...


class DiskContentStore:
    """JSON file-based content store with file locking."""

    def __init__(self, folder: Path, key: str) -> None:
        self._folder = folder
        self._key = key

    @property
    def _file(self) -> Path:
        return self._folder / f"{self._key}.json"

    def exists(self) -> bool:
        return self._file.exists()

    def read(self) -> dict | None:
        data, bad_format = None, False
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except ValueError:
            bad_format = True
        except OSError:
            LOGGER.debug("failed to read %s", self._file, exc_info=True)
        else:
            LOGGER.debug("got python info from %s", self._file)
            return data
        if bad_format:
            with suppress(OSError):
                self.remove()
        return None

    def write(self, content: dict) -> None:
        self._folder.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(content, sort_keys=True, indent=2), encoding="utf-8"
        )
        LOGGER.debug("wrote python info at %s", self._file)

    def remove(self) -> None:
        with suppress(OSError):
            self._file.unlink()
        LOGGER.debug("removed python info at %s", self._file)

    @contextmanager
    def locked(self) -> Generator[None]:
        from filelock import FileLock

        lock_path = self._folder / f"{self._key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path)):
            yield


class DiskCache:
    """File-system based Python interpreter info cache.

    Layout: ``<root>/py_info/4/<sha256(str(path))>.json`` -- identical to virtualenv's ``AppDataDiskFolder``.

    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def _py_info_dir(self) -> Path:
        return self._root / "py_info" / "4"

    def py_info(self, path: Path) -> DiskContentStore:
        key = sha256(str(path).encode("utf-8")).hexdigest()
        return DiskContentStore(self._py_info_dir, key)

    def py_info_clear(self) -> None:
        folder = self._py_info_dir
        if folder.exists():
            for f in folder.iterdir():
                if f.suffix == ".json":
                    with suppress(OSError):
                        f.unlink()


class NoOpContentStore:
    """Content store that does nothing."""

    def exists(self) -> bool:
        return False

    def read(self) -> dict | None:
        return None

    def write(self, content: dict) -> None:
        pass

    def remove(self) -> None:
        pass

    @contextmanager
    def locked(self) -> Generator[None]:
        yield


class NoOpCache:
    """Cache that does nothing -- used when caching is disabled."""

    def py_info(self, _path: Path) -> NoOpContentStore:
        return NoOpContentStore()

    def py_info_clear(self) -> None:
        pass


__all__ = [
    "ContentStore",
    "DiskCache",
    "DiskContentStore",
    "NoOpCache",
    "NoOpContentStore",
    "PyInfoCache",
]
