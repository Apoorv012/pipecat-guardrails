# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `OutputGuardProcessor`: a Pipecat processor that sits between the LLM and the TTS and checks
  each sentence before it is spoken. Allowed sentences pass, redacted ones are spoken with the
  matches replaced, and a block drops the rest of the reply and speaks the guard's fallback.
  Blocked text never reaches the TTS, so it never enters the conversation history.
- `strategy="full_reply"`: hold the whole reply, check it once, then speak it.
- Handles user interruptions, keeps non-text frames (function calls, scripted `TTSSpeakFrame`s)
  in order, checks `LLMTextProcessor` output as units, and skips expressive sound/pause units.
- `on_violation` and `on_check` events.
- `guarded_rtvi_observer_params()`: RTVI settings that stop raw, unchecked LLM text from being
  sent to client apps.
- Examples: `examples/pipeline_demo.py` (runs offline, no API keys) and `examples/pipecat_bot.py`
  (a full voice bot).

## [0.1.0] - 2026-10-31

First usable release: the rule layer and `Guard.check()`. The Pipecat processor and the
classifier / LLM-judge layers come in later releases.

### Added
- `Guard.check()`: checks text against the configured rules and returns a `Verdict` with the
  action (allow, flag, redact, block), the text to speak, violations, errors and per-rule latency.
- Rules (all opt-in): `BannedTerms` (whole-word, case- and Unicode-insensitive, works for
  Devanagari, with word variants), `Regex`, `PII`, and `@rule` for custom functions.
- PII recognizers: email, phone, card (Luhn), IP address, and custom regexes.
- India pack: Aadhaar (Verhoeff checksum), PAN, UPI id and Indian phone recognizers, plus
  `hinglish_word_pattern` for Hinglish spelling variants.
- Shadow mode, fail-open/fail-closed error policy, `enabled` flag and `Guard.enable()` /
  `Guard.disable()`, and an `on_decision` hook.
- Matching treats curly quotes and en/em dashes as ASCII, so `"don't worry"` also matches
  the curly-apostrophe form LLMs often write.
- Examples: the same health bot in English and Hinglish (`examples/whole_reply_*.py`).
- `SentenceChunker`: a streaming sentence splitter that handles decimals, abbreviations
  (including Indian ones like `Tab.` and `Smt.`), ellipses and Devanagari danda.

## [0.0.1] - 2026-10-31

### Added
- Package structure, CI and release workflow. No guardrail functionality yet.

[Unreleased]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Apoorv012/pipecat-guardrails/releases/tag/v0.0.1
