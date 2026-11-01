# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-11-02

Policies, answered by a classifier and an LLM judge, and an eval command to measure them.

### Added
- Policies: rules written as yes/no questions (`Guard(policies={"id": "Does this reply ...?"})`,
  or `Policy(...)` with examples of what counts as yes and no, thresholds, an action and an
  error policy). They catch what word lists miss: paraphrases, Hinglish, and safe sentences
  that happen to contain a banned word.
- The rules → classifier → judge cascade. `classifier=` (layer 2, e.g. TypeSafe Jev) answers
  every policy in one call and decides when it is sure; `judge=` (layer 3, any LLM through
  Pipecat's `LLMClassifier`) decides the rest. Either one can be used alone. Both take any
  Pipecat classifier.
- The classifier and judge see the redacted text, plus the last few turns of
  `CheckContext.history` (`history_turns=4`). Per-layer timeouts (`classifier_timeout=2.0`,
  `judge_timeout=5.0`); failures follow `on_error`, and a classifier failure goes to the judge.
- `OutputGuardProcessor` checks sentences concurrently and still releases them in order, so
  usually only the first sentence of a reply waits for a classifier or judge round trip.
  `max_parallel_checks` (default 5) caps how many run at once, as a sliding window: a
  sentence holds its slot until it is released, so after a block the checks still waiting
  are cancelled before they call the classifier or judge.
- `pipecat-guardrails eval data.jsonl --guard module:name`: precision, recall and F1 per label,
  latency per check and per layer, error rate, and the list of mistakes. Compare several
  guards at once, `--repeat`, `--json`, and `--diff` against a saved report to catch
  regressions in CI.
- A sample dataset and four guards (rules only, rules + Jev, rules + LLM judge, all three) in
  `examples/eval/`.
- Results on the sample in `docs/eval/`: adding the Jev classifier to the rules raised F1 from
  0.70 to 0.96 (recall 0.58 → 1.00) at about 310 ms per check.
- `Violation.layer` and `CheckError.layer` say which layer decided or failed;
  `CheckError.escalated` marks a classifier failure that went on to the judge.
- `Guard.enable()` / `Guard.disable()` also take policy ids; new `Guard.policies` and
  `Guard.policy()`.

### Changed
- `on_check` and `on_violation` fire when a sentence is released, in sentence order. A
  sentence dropped because an earlier one was blocked no longer fires them.
- If a check itself crashes (for example, a fallback function that raises), the processor
  now follows the guard's `on_error` instead of stopping.

## [0.2.1] - 2026-11-01

### Fixed
- `strategy="full_reply"`: a user interruption while a reply was held also threw away the
  function-call frames held with it. Pipecat marks those frames uninterruptible, and losing them
  left the function call out of the conversation history. They are now passed on.
- A plain `TextFrame` with `skip_tts=True` passed through unchecked. It is not spoken but can
  still reach the conversation history and the client app, so it is now checked: dropped if
  blocked, redacted if a rule redacts.

## [0.2.0] - 2026-10-31

The Pipecat processor: put one line between the LLM and the TTS.

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

[Unreleased]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Apoorv012/pipecat-guardrails/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Apoorv012/pipecat-guardrails/releases/tag/v0.0.1
