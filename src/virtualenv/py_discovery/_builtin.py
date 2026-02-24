from __future__ import annotations

import logging
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_data_path

from ._compat import IS_WIN, fs_path_id
from ._discover import Discover
from ._py_info import PythonInfo
from ._py_spec import PythonSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from ._cache import PyInfoCache

LOGGER = logging.getLogger(__name__)


class Builtin(Discover):
    python_spec: Sequence[str]
    cache: PyInfoCache | None
    try_first_with: Sequence[str]

    def __init__(
        self,
        python_spec: Sequence[str] | None = None,
        cache: PyInfoCache | None = None,
        try_first_with: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(env=env)
        self.python_spec = python_spec or [sys.executable]
        self.cache = cache
        self.try_first_with = try_first_with or []

    def run(self) -> PythonInfo | None:
        for python_spec in self.python_spec:
            result = get_interpreter(
                python_spec, self.try_first_with, self.cache, self._env
            )
            if result is not None:
                return result
        return None

    def __repr__(self) -> str:
        spec = self.python_spec[0] if len(self.python_spec) == 1 else self.python_spec
        return f"{self.__class__.__name__} discover of python_spec={spec!r}"


def get_interpreter(
    key,
    try_first_with: Iterable[str],
    cache: PyInfoCache | None = None,
    env: Mapping[str, str] | None = None,
) -> PythonInfo | None:
    spec = PythonSpec.from_string_spec(key)
    LOGGER.info("find interpreter for spec %r", spec)
    proposed_paths = set()
    env = os.environ if env is None else env
    for interpreter, impl_must_match in propose_interpreters(
        spec, try_first_with, cache, env
    ):
        key = interpreter.system_executable, impl_must_match
        if key in proposed_paths:
            continue
        LOGGER.info("proposed %s", interpreter)
        if interpreter.satisfies(spec, impl_must_match):
            LOGGER.debug("accepted %s", interpreter)
            return interpreter
        proposed_paths.add(key)
    return None


def propose_interpreters(
    spec: PythonSpec,
    try_first_with: Iterable[str],
    cache: PyInfoCache | None = None,
    env: Mapping[str, str] | None = None,
) -> Generator[tuple[PythonInfo, bool], None, None]:
    env = os.environ if env is None else env
    tested_exes: set[str] = set()
    if spec.is_abs and spec.path is not None:
        try:
            os.lstat(spec.path)
        except OSError:
            pass
        else:
            exe_raw = os.path.abspath(spec.path)
            exe_id = fs_path_id(exe_raw)
            if exe_id not in tested_exes:
                tested_exes.add(exe_id)
                yield PythonInfo.from_exe(exe_raw, cache, env=env), True
        return

    for py_exe in try_first_with:
        path = os.path.abspath(py_exe)
        try:
            os.lstat(path)
        except OSError:
            pass
        else:
            exe_raw = os.path.abspath(path)
            exe_id = fs_path_id(exe_raw)
            if exe_id in tested_exes:
                continue
            tested_exes.add(exe_id)
            yield PythonInfo.from_exe(exe_raw, cache, env=env), True

    if spec.path is not None:
        try:
            os.lstat(spec.path)
        except OSError:
            pass
        else:
            exe_raw = os.path.abspath(spec.path)
            exe_id = fs_path_id(exe_raw)
            if exe_id not in tested_exes:
                tested_exes.add(exe_id)
                yield PythonInfo.from_exe(exe_raw, cache, env=env), True
        if spec.is_abs:
            return
    else:
        current_python = PythonInfo.current_system(cache)
        exe_raw = str(current_python.executable)
        exe_id = fs_path_id(exe_raw)
        if exe_id not in tested_exes:
            tested_exes.add(exe_id)
            yield current_python, True

        if IS_WIN:
            from ._windows import propose_interpreters

            for interpreter in propose_interpreters(spec, cache, env):
                exe_raw = str(interpreter.executable)
                exe_id = fs_path_id(exe_raw)
                if exe_id in tested_exes:
                    continue
                tested_exes.add(exe_id)
                yield interpreter, True

    find_candidates = path_exe_finder(spec)
    for pos, path in enumerate(get_paths(env)):
        LOGGER.debug(LazyPathDump(pos, path, env))
        for exe, impl_must_match in find_candidates(path):
            exe_raw = str(exe)
            if resolved := _resolve_shim(exe_raw, env):
                LOGGER.debug("resolved shim %s to %s", exe_raw, resolved)
                exe_raw = resolved
            exe_id = fs_path_id(exe_raw)
            if exe_id in tested_exes:
                continue
            tested_exes.add(exe_id)
            interpreter = PathPythonInfo.from_exe(
                exe_raw, cache, raise_on_error=False, env=env
            )
            if interpreter is not None:
                yield interpreter, impl_must_match

    if uv_python_dir := os.getenv("UV_PYTHON_INSTALL_DIR"):
        uv_python_path = Path(uv_python_dir).expanduser()
    elif xdg_data_home := os.getenv("XDG_DATA_HOME"):
        uv_python_path = Path(xdg_data_home).expanduser() / "uv" / "python"
    else:
        uv_python_path = user_data_path("uv") / "python"

    for exe_path in uv_python_path.glob("*/bin/python"):
        interpreter = PathPythonInfo.from_exe(
            str(exe_path), cache, raise_on_error=False, env=env
        )
        if interpreter is not None:
            yield interpreter, True


def get_paths(env: Mapping[str, str]) -> Generator[Path, None, None]:
    path = env.get("PATH", None)
    if path is None:
        try:
            path = os.confstr("CS_PATH")
        except (AttributeError, ValueError):
            path = os.defpath
    if path:
        for p in map(Path, path.split(os.pathsep)):
            with suppress(OSError):
                if p.is_dir() and next(p.iterdir(), None):
                    yield p


class LazyPathDump:
    def __init__(self, pos: int, path: Path, env: Mapping[str, str]) -> None:
        self.pos = pos
        self.path = path
        self.env = env

    def __repr__(self) -> str:
        content = f"discover PATH[{self.pos}]={self.path}"
        if self.env.get("_VIRTUALENV_DEBUG"):
            content += " with =>"
            for file_path in self.path.iterdir():
                try:
                    if file_path.is_dir():
                        continue
                    if IS_WIN:
                        pathext = self.env.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(
                            ";"
                        )
                        if not any(
                            file_path.name.upper().endswith(ext) for ext in pathext
                        ):
                            continue
                    elif not (file_path.stat().st_mode & os.X_OK):
                        continue
                except OSError:
                    pass
                content += " "
                content += file_path.name
        return content


def path_exe_finder(
    spec: PythonSpec,
) -> Callable[[Path], Generator[tuple[Path, bool], None, None]]:
    """Given a spec, return a function that can be called on a path to find all matching files in it."""
    pat = spec.generate_re(windows=sys.platform == "win32")
    direct = spec.str_spec
    if sys.platform == "win32":
        direct = f"{direct}.exe"

    def path_exes(path: Path) -> Generator[tuple[Path, bool], None, None]:
        direct_path = path / direct
        if direct_path.exists():
            yield direct_path, False

        for exe in path.iterdir():
            match = pat.fullmatch(exe.name)
            if match:
                yield exe.absolute(), match["impl"] == "python"

    return path_exes


def _resolve_shim(exe_path: str, env: Mapping[str, str]) -> str | None:
    """Resolve a version-manager shim to the actual Python binary."""
    for shims_dir_env, versions_path in _VERSION_MANAGER_LAYOUTS:
        if root := env.get(shims_dir_env):
            shims_dir = os.path.join(root, "shims")
            if os.path.dirname(exe_path) == shims_dir:
                exe_name = os.path.basename(exe_path)
                versions_dir = os.path.join(root, *versions_path)
                return _resolve_shim_to_binary(exe_name, versions_dir, env)
    return None


_VERSION_MANAGER_LAYOUTS: list[tuple[str, tuple[str, ...]]] = [
    ("PYENV_ROOT", ("versions",)),
    ("MISE_DATA_DIR", ("installs", "python")),
    ("ASDF_DATA_DIR", ("installs", "python")),
]


def _resolve_shim_to_binary(
    exe_name: str, versions_dir: str, env: Mapping[str, str]
) -> str | None:
    for version in _active_versions(env):
        resolved = os.path.join(versions_dir, version, "bin", exe_name)
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
    return None


def _active_versions(env: Mapping[str, str]) -> Generator[str, None, None]:
    """Yield active Python version strings by reading version-manager configuration."""
    if pyenv_version := env.get("PYENV_VERSION"):
        yield from pyenv_version.split(":")
        return
    if versions := _read_python_version_file(os.getcwd()):
        yield from versions
        return
    if (pyenv_root := env.get("PYENV_ROOT")) and (
        versions := _read_python_version_file(
            os.path.join(pyenv_root, "version"), search_parents=False
        )
    ):
        yield from versions


def _read_python_version_file(
    start: str, *, search_parents: bool = True
) -> list[str] | None:
    """Read a ``.python-version`` file, optionally searching parent directories."""
    current = start
    while True:
        candidate = (
            os.path.join(current, ".python-version")
            if os.path.isdir(current)
            else current
        )
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as f:
                if versions := [
                    v for line in f if (v := line.strip()) and not v.startswith("#")
                ]:
                    return versions
        if not search_parents:
            return None
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


class PathPythonInfo(PythonInfo):
    """python info from path."""


__all__ = [
    "Builtin",
    "LazyPathDump",
    "PathPythonInfo",
    "get_interpreter",
    "get_paths",
    "propose_interpreters",
]
