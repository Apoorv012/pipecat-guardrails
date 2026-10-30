"""Output guardrails for Pipecat voice agents.

Checks what a voice agent is about to say and stops it before it is spoken.
This package is in early development; see the README for status.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pipecat-guardrails")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = ["__version__"]
