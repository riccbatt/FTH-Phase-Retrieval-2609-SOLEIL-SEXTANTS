"""Small, dependency-light setup helpers shared by the notebooks."""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings

from IPython import get_ipython


_QT_BINDINGS = (
    ("pyqt5", "PyQt5.QtCore", "qt5"),
    ("pyside6", "PySide6.QtCore", "qt6"),
    ("pyqt6", "PyQt6.QtCore", "qt6"),
    ("pyside2", "PySide2.QtCore", "qt5"),
)


def _available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _already_loaded_qt_api() -> str | None:
    for qt_api, module_name, _ in _QT_BINDINGS:
        package = module_name.split(".", 1)[0]
        if package in sys.modules:
            return qt_api
    return None


def configure_matplotlib_qt(fallback: str = "inline") -> str:
    """Select one working Qt binding before pyplot or GUI libraries import.

    PyQt5 is preferred because it is the binding used on the SOLEIL analysis
    machine. If a different binding has already been imported, the function
    uses that binding instead of attempting an unsafe in-process switch.
    When Qt cannot start, it falls back without aborting the notebook.
    """
    shell = get_ipython()
    if shell is None:
        return "non-IPython"

    loaded_api = _already_loaded_qt_api()
    candidates = list(_QT_BINDINGS)
    if loaded_api is not None:
        candidates.sort(key=lambda item: item[0] != loaded_api)

    errors = []
    for qt_api, module_name, gui_name in candidates:
        if not _available(module_name):
            continue
        if loaded_api is not None and qt_api != loaded_api:
            continue
        os.environ["QT_API"] = qt_api
        try:
            shell.run_line_magic("matplotlib", gui_name)
            return f"{gui_name} ({qt_api})"
        except Exception as exc:  # IPython/Qt versions raise several exception types.
            errors.append(f"{qt_api}: {exc}")

    message = "Could not start a Qt Matplotlib backend."
    if loaded_api is not None:
        message += (
            f" Qt binding {loaded_api!r} was already imported. Restart the kernel "
            "and run the first cell before importing matplotlib, napari, PyQt, or PySide."
        )
    if errors:
        message += " " + " | ".join(errors)
    warnings.warn(f"{message} Falling back to %{fallback}.", RuntimeWarning)
    shell.run_line_magic("matplotlib", fallback)
    return fallback
