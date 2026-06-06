"""Ambiente de subprocesso compatível com Windows (stdout UTF-8)."""

from __future__ import annotations

import os
import sys


def configure_stdio_utf8() -> None:
    """Evita ``UnicodeEncodeError`` no console Windows (cp1252)."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env
