from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import pytest

from virtualenv.activation import BatchActivator


def test_batch_pydoc_bat_quoting(tmp_path) -> None:
    """Test that pydoc.bat properly quotes python.exe path to handle spaces."""

    # GIVEN: A mock interpreter
    class MockInterpreter:
        os = "nt"
        tcl_lib = None
        tk_lib = None

    class MockCreator:
        def __init__(self, dest) -> None:
            self.dest = dest
            self.bin_dir = dest / "Scripts"
            self.bin_dir.mkdir(parents=True)
            self.interpreter = MockInterpreter()
            self.pyenv_cfg = {}
            self.env_name = "test-env"

    creator = MockCreator(tmp_path)
    options = Namespace(prompt=None)
    activator = BatchActivator(options)

    # WHEN: Generate activation scripts
    activator.generate(creator)

    # THEN: pydoc.bat should quote python.exe to handle paths with spaces
    pydoc_content = (creator.bin_dir / "pydoc.bat").read_text(encoding="utf-8")

    # The python.exe should be quoted to handle paths with spaces like "C:\Program Files\Python39\python.exe"
    assert '"python.exe"' in pydoc_content, f"python.exe should be quoted in pydoc.bat. Content:\n{pydoc_content}"


@pytest.mark.parametrize(
    ("tcl_lib", "tk_lib", "present"),
    [
        ("C:\\tcl", "C:\\tk", True),
        (None, None, False),
    ],
)
def test_batch_tkinter_generation(tmp_path, tcl_lib, tk_lib, present) -> None:
    # GIVEN
    class MockInterpreter:
        os = "nt"

    interpreter = MockInterpreter()
    interpreter.tcl_lib = tcl_lib
    interpreter.tk_lib = tk_lib

    class MockCreator:
        def __init__(self, dest) -> None:
            self.dest = dest
            self.bin_dir = dest / "bin"
            self.bin_dir.mkdir()
            self.interpreter = interpreter
            self.pyenv_cfg = {}
            self.env_name = "my-env"

    creator = MockCreator(tmp_path)
    options = Namespace(prompt=None)
    activator = BatchActivator(options)

    # WHEN
    activator.generate(creator)
    activate_content = (creator.bin_dir / "activate.bat").read_text(encoding="utf-8")
    deactivate_content = (creator.bin_dir / "deactivate.bat").read_text(encoding="utf-8")

    # THEN
    # PKG_CONFIG_PATH is always set
    assert '@if defined PKG_CONFIG_PATH @set "_OLD_PKG_CONFIG_PATH=%PKG_CONFIG_PATH%"' in activate_content
    assert '@set "PKG_CONFIG_PATH=%VIRTUAL_ENV%\\lib\\pkgconfig;%PKG_CONFIG_PATH%"' in activate_content
    assert '@if defined _OLD_PKG_CONFIG_PATH @set "PKG_CONFIG_PATH=%_OLD_PKG_CONFIG_PATH%"' in deactivate_content
    assert "@if not defined _OLD_PKG_CONFIG_PATH @set PKG_CONFIG_PATH=" in deactivate_content
    assert "@set _OLD_PKG_CONFIG_PATH=" in deactivate_content

    if present:
        assert '@if NOT "C:\\tcl"=="" @set "TCL_LIBRARY=C:\\tcl"' in activate_content
        assert '@if NOT "C:\\tk"=="" @set "TK_LIBRARY=C:\\tk"' in activate_content
        assert "if defined _OLD_VIRTUAL_TCL_LIBRARY" in deactivate_content
        assert "if defined _OLD_VIRTUAL_TK_LIBRARY" in deactivate_content
    else:
        assert '@if NOT ""=="" @set "TCL_LIBRARY="' in activate_content
        assert '@if NOT ""=="" @set "TK_LIBRARY="' in activate_content


@pytest.mark.usefixtures("activation_python")
@pytest.mark.parametrize("activations", [1, 2], ids=["activate_once", "activate_twice"])
def test_batch(activation_tester_class, activation_tester, tmp_path, activations) -> None:
    version_script = tmp_path / "version.bat"
    version_script.write_text("ver", encoding="utf-8")

    class Batch(activation_tester_class):
        def __init__(self, session) -> None:
            super().__init__(BatchActivator, session, None, "activate.bat", "bat")
            self._version_cmd = [str(version_script)]
            self._invoke_script = []
            self.deactivate = "call deactivate"
            self.activate_cmd = "call"
            self.pydoc_call = f"call {self.pydoc_call}"
            self.unix_line_ending = False

        def _get_test_lines(self, activate_script):
            return ["@echo off", *super()._get_test_lines(activate_script)]

        def activate_call(self, script):
            # activating again without deactivating must still restore the values from before the first activation
            return " & ".join([super().activate_call(script)] * activations)

        def quote(self, s):
            if '"' in s or " " in s:
                text = s.replace('"', r"\"")
                return f'"{text}"'
            return s

        def print_prompt(self) -> str:
            return 'echo "%PROMPT%"'

        def assert_pkg_config_path(self, before, activated, deactivated, raw) -> None:
            # cmd.exe has no way to represent &|<>()^ literally inside @set "VAR=value" (confirmed on a
            # real Windows runner - Microsoft's own caret escape is suppressed inside the quotes), so
            # BatchActivator.quote() neuters them. special_char_name carries several of these, so the
            # PKG_CONFIG_PATH entry activate.bat actually sets is built from the neutered dest, not the
            # raw one the base assertion compares against.
            user_value = os.environ.get("PKG_CONFIG_PATH")
            neutered_dest = BatchActivator.quote(str(self._creator.dest))
            assert (before, [self.norm_path(entry) for entry in activated.split(os.pathsep)], deactivated) == (
                str(user_value),
                [
                    self.norm_path(Path(neutered_dest) / "lib" / "pkgconfig"),
                    *([self.norm_path(user_value)] if user_value else []),
                ],
                str(user_value),
            ), raw

    activation_tester(Batch)


@pytest.mark.usefixtures("activation_python")
def test_batch_output(activation_tester_class, activation_tester, tmp_path) -> None:
    version_script = tmp_path / "version.bat"
    version_script.write_text("ver", encoding="utf-8")

    class Batch(activation_tester_class):
        def __init__(self, session) -> None:
            super().__init__(BatchActivator, session, None, "activate.bat", "bat")
            self._version_cmd = [str(version_script)]
            self._invoke_script = []
            self.deactivate = "call deactivate"
            self.activate_cmd = "call"
            self.pydoc_call = f"call {self.pydoc_call}"
            self.unix_line_ending = False

        def _get_test_lines(self, activate_script):
            """Build intermediary script which will be then called. In the script just activate environment, call echo to get current echo setting, and then deactivate. This ensures that echo setting is preserved and no unwanted output appears."""
            intermediary_script_path = str(tmp_path / "intermediary.bat")
            activate_script_quoted = self.quote(str(activate_script))
            return [
                "@echo on",
                f"@echo @call {activate_script_quoted} > {intermediary_script_path}",
                f"@echo @echo >> {intermediary_script_path}",
                f"@echo @deactivate >> {intermediary_script_path}",
                f"@call {intermediary_script_path}",
            ]

        def assert_output(self, out, raw, tmp_path) -> None:  # ruff:ignore[unused-method-argument]
            assert out[0] == "ECHO is on.", raw

        def quote(self, s):
            if '"' in s or " " in s:
                text = s.replace('"', r"\"")
                return f'"{text}"'
            return s

        def print_prompt(self) -> str:
            return 'echo "%PROMPT%"'

    activation_tester(Batch)
