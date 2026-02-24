from __future__ import annotations

import sys

import pytest

from virtualenv.py_discovery import DiskCache, PythonInfo


@pytest.fixture(scope="session")
def session_cache(tmp_path_factory: pytest.TempPathFactory) -> DiskCache:
    return DiskCache(tmp_path_factory.mktemp("py-discovery-cache"))


@pytest.fixture(autouse=True)
def _ensure_py_info_cache_empty(session_cache: DiskCache) -> None:
    PythonInfo.clear_cache(session_cache)
    yield
    PythonInfo.clear_cache(session_cache)


@pytest.fixture
def _skip_if_test_in_system(session_cache: DiskCache) -> None:
    current = PythonInfo.current(session_cache)
    if current.system_executable is not None:
        pytest.skip("test not valid if run under system")


@pytest.fixture(scope="session")
def for_py_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"
