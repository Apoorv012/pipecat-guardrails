# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Apoorv012/pipecat-guardrails/releases/tag/v0.0.1
