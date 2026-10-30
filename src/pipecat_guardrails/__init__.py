"""Output guardrails for Pipecat voice agents.

Checks what a voice agent is about to say and stops it before it is spoken. Every check is
opt-in: configure only the rules you want.

Example::

    from pipecat_guardrails import Guard
    from pipecat_guardrails.rules import BannedTerms

    guard = Guard(
        rules=[BannedTerms(["acme corp"], id="no-competitors")],
        fallback="Sorry, I can't help with that.",
    )
    verdict = await guard.check("Have you tried Acme Corp?")
"""

from importlib.metadata import PackageNotFoundError, version

from pipecat_guardrails.guard import Guard
from pipecat_guardrails.types import (
    Action,
    CheckContext,
    CheckError,
    Verdict,
    Violation,
)

try:
    __version__ = version("pipecat-guardrails")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "Action",
    "CheckContext",
    "CheckError",
    "Guard",
    "Verdict",
    "Violation",
    "__version__",
]
