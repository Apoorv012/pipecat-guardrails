"""Layer 1: local rules (banned terms, regex, personal data, custom functions).

Every rule is opt-in: a guard only runs the rules you give it.
"""

from pipecat_guardrails.rules import pii
from pipecat_guardrails.rules.base import FunctionRule, Match, Rule, rule
from pipecat_guardrails.rules.pii import PII, Recognizer
from pipecat_guardrails.rules.regex import Regex
from pipecat_guardrails.rules.terms import BannedTerms

__all__ = [
    "PII",
    "BannedTerms",
    "FunctionRule",
    "Match",
    "Recognizer",
    "Regex",
    "Rule",
    "pii",
    "rule",
]
