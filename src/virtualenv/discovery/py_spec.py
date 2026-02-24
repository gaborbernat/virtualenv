"""Backward-compatible re-export -- external tools import from here."""

from __future__ import annotations

from virtualenv.py_discovery._py_spec import PythonSpec

__all__ = [
    "PythonSpec",
]
