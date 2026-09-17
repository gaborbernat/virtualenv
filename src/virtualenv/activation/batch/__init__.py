from __future__ import annotations

import os
from typing import TYPE_CHECKING

from virtualenv.activation.via_template import ViaTemplateActivator
from virtualenv.util.text import collapse_line_boundaries

if TYPE_CHECKING:
    from collections.abc import Iterator

    from python_discovery import PythonInfo

    from virtualenv.create.creator import Creator


class BatchActivator(ViaTemplateActivator):
    @classmethod
    def supports(cls, interpreter: PythonInfo) -> bool:
        return interpreter.os == "nt"

    def templates(self) -> Iterator[str]:
        yield "activate.bat"
        yield "deactivate.bat"
        yield "pydoc.bat"

    @staticmethod
    def quote(string: str) -> str:
        """Make a value safe to sit inside ``@set "VAR=value"``.

        Batch has no escape for a double quote in this context: it always closes the quoted string,
        and whatever follows on the line runs as live cmd.exe syntax (``&``, ``|``, redirections, ...).
        A raw line boundary is worse - batch is line-oriented regardless of quote state, so it starts
        a brand-new statement instead of staying inside the value. Neither can be represented
        literally here, so replace them with a space. ``%`` still triggers variable expansion inside
        the quotes, but doubling it to ``%%`` is a real, in-file escape that keeps the literal
        character.
        """
        return collapse_line_boundaries(string.replace("%", "%%").replace('"', " "))

    def instantiate_template(self, replacements: dict[str, str], template: str, creator: Creator) -> str:
        # ensure the text has all newlines as \r\n - required by batch
        base = super().instantiate_template(replacements, template, creator)
        return base.replace(os.linesep, "\n").replace("\n", os.linesep)


__all__ = [
    "BatchActivator",
]
