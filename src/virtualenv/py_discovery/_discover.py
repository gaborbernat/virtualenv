"""Abstract base class for Python interpreter discovery mechanisms."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._py_info import PythonInfo


class Discover(ABC):
    """Discover and provide the requested Python interpreter."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._has_run = False
        self._interpreter: PythonInfo | None = None
        self._env = env if env is not None else os.environ

    @abstractmethod
    def run(self) -> PythonInfo | None:
        """Discover an interpreter.

        :returns: the interpreter ready to use, or ``None`` if not found

        """
        raise NotImplementedError

    @property
    def interpreter(self) -> PythonInfo | None:
        """:returns: the interpreter as returned by :meth:`run`, cached"""
        if self._has_run is False:
            self._interpreter = self.run()
            self._has_run = True
        return self._interpreter


__all__ = [
    "Discover",
]
