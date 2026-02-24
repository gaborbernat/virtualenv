"""Virtualenv-specific Discover that wraps py_discovery's base class with VirtualEnvOptions."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from virtualenv.py_discovery import Discover as _Discover

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from virtualenv.config.cli.parser import VirtualEnvOptions
    from virtualenv.py_discovery import PythonInfo


class Discover(_Discover):
    @classmethod
    def add_parser_arguments(cls, parser: ArgumentParser) -> None:
        raise NotImplementedError

    def __init__(self, options: VirtualEnvOptions) -> None:
        super().__init__(env=options.env)

    @abstractmethod
    def run(self) -> PythonInfo | None:
        raise NotImplementedError


__all__ = [
    "Discover",
]
