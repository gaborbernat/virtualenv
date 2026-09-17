from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final

from virtualenv.activation.via_template import ViaTemplateActivator
from virtualenv.util.text import collapse_line_boundaries

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from python_discovery import PythonInfo

    from virtualenv.create.creator import Creator

# cmd.exe's own command-line parser looks for these regardless of surrounding quotes: confirmed on a
# real Windows runner that `@set "VAR=x & cmd"` still runs `cmd` as a second statement even though the
# whole VAR=value expression is quoted. Microsoft's documented fix is to escape them with the caret
# (^), but caret escaping is itself suppressed inside a quoted string - the exact context every value
# here is substituted into - so there is no way to keep them literal.
_CMD_OPERATORS: Final[tuple[str, ...]] = ("&", "|", "<", ">", "(", ")", "^", '"')

# Filesystem paths, unlike the prompt, must keep pointing at the real directory or activation silently
# breaks: %VIRTUAL_ENV%\Scripts stops resolving, so deactivate and the venv's own python vanish from
# PATH with no error. These are the replacement keys that name an existing directory rather than free
# text, so they get a real escape (an 8.3 short name) instead of having their unsafe characters
# neutered.
_PATH_REPLACEMENT_KEYS: Final[tuple[str, ...]] = ("__VIRTUAL_ENV__", "__TCL_LIBRARY__", "__TK_LIBRARY__")


def _needs_short_path(value: str) -> bool:
    return any(char in value for char in _CMD_OPERATORS) or collapse_line_boundaries(value) != value


def _short_path(path: str) -> str | None:
    """Windows 8.3 short name for `path`, immune to every character `quote()` would otherwise neuter.

    Requires the path to already exist and 8.3 name generation to be enabled for its volume (the
    default). Returns ``None`` when either does not hold, or when not actually running on Windows -
    ``quote()`` remains the fallback, so a value that cannot be shortened is merely neutered rather than
    left able to break out of the ``@set "VAR=value"`` construct.
    """
    if sys.platform != "win32":  # pragma: win32 cover
        return None
    from ctypes import create_unicode_buffer, windll  # ruff:ignore[import-outside-top-level]

    buffer = create_unicode_buffer(260)
    n = windll.kernel32.GetShortPathNameW(path, buffer, len(buffer))
    import sys as _s

    print(f"DEBUG _short_path({path!r}) -> n={n} value={buffer.value!r}", file=_s.stderr)  # noqa
    if n == 0:
        return None
    return buffer.value or None


class BatchActivator(ViaTemplateActivator):
    @classmethod
    def supports(cls, interpreter: PythonInfo) -> bool:
        return interpreter.os == "nt"

    def templates(self) -> Iterator[str]:
        yield "activate.bat"
        yield "deactivate.bat"
        yield "pydoc.bat"

    def replacements(self, creator: Creator, dest_folder: Path) -> dict[str, str]:
        values = super().replacements(creator, dest_folder)
        for key in _PATH_REPLACEMENT_KEYS:
            if (value := values[key]) and _needs_short_path(value) and (short := _short_path(value)):
                values[key] = short
        return values

    @staticmethod
    def quote(string: str) -> str:
        """Make a value safe to sit inside ``@set "VAR=value"``.

        Batch has no escape for a double quote in this context either: it always closes the quoted
        string, and whatever follows on the line runs as live cmd.exe syntax. A raw line boundary is
        worse - batch is line-oriented regardless of quote state, so it starts a brand-new statement
        instead of staying inside the value. None of these can be represented literally here, so
        replace them with a space. ``%`` still triggers variable expansion inside the quotes, but
        doubling it to ``%%`` is a real, in-file escape that keeps the literal character.
        """
        string = string.replace("%", "%%")
        for operator in _CMD_OPERATORS:
            string = string.replace(operator, " ")
        return collapse_line_boundaries(string)

    def instantiate_template(self, replacements: dict[str, str], template: str, creator: Creator) -> str:
        # ensure the text has all newlines as \r\n - required by batch
        base = super().instantiate_template(replacements, template, creator)
        return base.replace(os.linesep, "\n").replace("\n", os.linesep)


__all__ = [
    "BatchActivator",
]
