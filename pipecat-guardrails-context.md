# pipecat-guardrails — project context

Handoff notes for starting this project. Everything below was researched or checked on 2026-10-30;
items marked **unverified** were not confirmed against source.

Working name: **`pipecat-guardrails`** (import `pipecat_guardrails`). Free on PyPI as of 2026-10-30, as
are `pipecat-guard`, `pipecat-output-guard`, `pipecat-rails`, `pipecat-shield`, `pipecat-safety`,
`pipecat-moderation`.

## 1. What this is

An **output guardrail** library for voice agents, with Pipecat as the first-class integration. It sits
between the LLM and the TTS and checks what the agent is about to say against policies. On a violation it
stops the speech, not just the text: cancel the LLM, flush queued audio, play a scripted recovery line,
and keep the blocked text out of the conversation history.

It also has to work without Pipecat: a plain `check(text)` call for apps that run their own LLM loop
and only use Pipecat for audio. The first consumer, VoiceMitra, does exactly that (section 9).

## 2. Why it's new (and what NOT to rebuild)

Most parts of a guardrail exist already. The gap is the voice-side enforcement and Indic support.

| Already exists | Where | Use it, don't rebuild it |
|---|---|---|
| Streaming output rails | NeMo Guardrails (token chunks, default 200 tokens, `stream_first` option) | Prior art |
| Validator framework | **Guardrails AI** (`guardrails-ai` 0.11.0, Apache-2.0, py>=3.10,<3.14, released 2026-08-14). `Guard().use(validator)`; `guard.validate("text")` checks an existing string with no LLM call; streaming (`stream=True`, sentence chunks per validator, async supported) only when the Guard itself calls the LLM, and "not all on_fail types are supported with streaming"; `guard.error_spans_in_output()`; server mode (`guardrails start`, Flask, OpenAI-compatible route). Hub of ~65 validators, e.g. CompetitorCheck, DetectPII (Presidio, English-centric), ToxicLanguage (BERT). Validators moving to plain PyPI packages; hosted remote inference discontinued 2026-08-25. Heavy core deps (litellm, langchain-core, opentelemetry, `pip>=26`). **No voice, Pipecat or LiveKit integration found** | Optional rail backend via an adapter (section 3), never a core dependency. Its streaming assumes it owns the LLM call; in Pipecat the LLM service does, so only per-sentence `validate()` applies, and the voice enforcement around it is still missing |
| Safety classifiers | Llama Guard 3 (Hindi officially supported), ShieldGemma, Qwen3Guard (119 languages claimed; 0.6B variant), Granite Guardian (English-only per one comparison) | Optional backends |
| Decision models (typed yes/no, choice, score) | TypeSafe **Jev** (hosted, closed); open: **Laya** (421M ModernBERT, Apache 2.0, 100+ languages), **Kev** (Qwen-based, 0.8B+), Maya (435M, **unverified**, one Medium post only) | Judge backends |
| A common classifier interface | Pipecat `pipecat.classifiers.base_classifier.BaseClassifier` with `ask()`, `yes_no()`, `choice()`, `score()`; backends `JevClassifier`, `LLMClassifier` (in 1.12), `ClefClassifier` and `OpenAIDecisionsClassifier` (added in 1.13) | **Adapt to it**, so every Pipecat classifier becomes a judge for free |
| Rewriting LLM text before TTS | Pipecat 1.14 `LLMTextProcessor.text_transforms` / `add_text_transformer()`. TTS `text_filters` / `BaseTextFilter` are deprecated as of 1.14 | Rewrite only; can't block, cut audio or keep text out of history |
| Spend/budget limit | `floe-guard` 0.25.2 (Floe Labs): meters LLM/STT/TTS/telephony cost per call and hard-stops the next turn before a spend ceiling; has a `pipecat` extra (`pipecat-ai>=1.0`); listed in Pipecat docs under analytics (https://docs.pipecat.ai/api-reference/server/services/analytics/floe) | Different problem (cost, not content); shows the docs list community packages |
| Input-side protection | `pipecat-firewall` 0.1.0 (prompt injection, PII extraction, policy override; placed after STT). Requires `pipecat-ai<1.0,>=0.0.98`, so it doesn't run on Pipecat 1.x | Different problem (input) |

Pipecat has **no output guardrail processor** as of 1.15.0 (2026-10-28). Checked: the installed 1.12.0
source, and the 1.13 to 1.15 CHANGELOG entries for guardrail, moderation, safety, classifier and text
filter.

What this project adds:
1. **Voice enforcement**: the full violation sequence inside a Pipecat pipeline (section 5).
2. **Latency-budgeted tiers**: fast blocking rails per sentence before TTS; a slow judge either blocking
   or running alongside playback and cutting audio after the fact.
3. **Indic-aware fast rails**: Hinglish and Devanagari term matching with transliteration variants
   (dawai / davai / दवाई), plus Indian PII (Aadhaar with Verhoeff checksum, PAN, +91 phone, UPI ID).
   Presidio-style tools are English-centric.
4. **A benchmark harness** for voice guardrails: labelled replies, scored per rail or judge on
   precision, recall and latency (section 7).

## 3. Architecture

A core with no framework imports, plus thin adapters:

```
pipecat_guardrails/
  core (no pipecat import)
    types        Verdict(action=allow|rewrite|block, rule, reason, fallback_text, latency_ms, scores)
    chunker      sentence splitter for streamed text
    rails/
      terms      exact-term rail
      pii        Indian PII rail (+ optional Presidio backend)
      judge      "Decider" protocol: yes/no policy questions -> calibrated probability
    guard        Guard (whole reply) + StreamGuard (token stream -> approved sentences / violation)
    log          decision log (every trigger, for shadow mode and evals)
  adapters/
    pipecat_processor   OutputGuardProcessor(FrameProcessor), between llm and tts
    pipecat_classifier  wraps any pipecat BaseClassifier as a Decider (Jev, LLM, Clef, OpenAI Decisions)
    openai_judge        policy-prompted judge over any OpenAI-compatible chat API (JSON verdict)
    hf_local            optional transformers backend (Qwen3Guard / Llama Guard / Laya), lazy import
    guardrails_ai       wraps any Guardrails AI validator as a rail (calls validator per sentence; optional extra)
  bench/
    run a labelled JSONL set through each rail or judge; report P/R/F1 + p50/p95 latency
```

Heavy dependencies (transformers, presidio, pipecat) go in **optional extras** and are imported lazily.
The core should install with nothing heavy.

### Guard behaviour
- **Tiers**: `fast` (blocking, about 1 ms per sentence; terms, PII, regex), `slow_blocking` (judge, must finish
  before TTS), `background` (judge runs while audio plays; a violation cuts the audio).
- **Modes**: `enforce`, or `shadow` (log only; for introducing new rules safely).
- **Per-rule action**: block (with fallback text), rewrite/redact, or flag.
- **Judge failure policy**: timeout per judge, then fail open or fail closed, configurable per rule.
- **Two APIs**: `Guard.check(text, context) -> Verdict` for a whole reply, and `StreamGuard` fed tokens.

### Chunker requirements
Split on `.` `?` `!` `।` `॥` and newlines. Don't split on abbreviations (`Dr.`, `Mr.`, `mg.`), decimals
(`2.5 mg`), times (`8.30`), or ellipses mid-sentence. Flush the remainder at end of reply. Reset on interruption.

## 4. Judges: how to pick one
Answer "which judge" with benchmark data, not opinion. Candidates:
1. Rules only (baseline).
2. **Jev** via `JevClassifier`: policies map directly to yes/no questions ("Does this reply advise changing
   a medicine dose?"). Hosted; default model `jev-1.13.0`, endpoint `https://api.typesafe.ai`; needs a
   TypeSafe API key (early access since 2026-09-15). Calibrated probabilities, so one threshold works
   across questions.
3. **Gemini Flash-Lite** with a policy prompt (OpenAI-compatible endpoint).
4. **Sarvam** (`sarvam-105b-conversations`) with a policy prompt: likely best at Hinglish, likely slower.
5. **Laya** locally: test whether a 421M encoder is fast enough on CPU.

Expectation going in (to be confirmed or overturned): generic safety classifiers (Llama Guard and similar) target
harms like violence and self-harm. They'll miss domain rules like "dawai double kar lijiye" or "chinta mat
kijiye", or over-flag any medical talk. Policy-specific judges (decision models, or an LLM with a policy
prompt) should win on domain policies.

## 5. Pipecat adapter: OutputGuardProcessor

Placement:
```python
Pipeline([
    transport.input(), stt, context_aggregator.user(), llm,
    OutputGuardProcessor(guard),   # here
    tts, transport.output(), context_aggregator.assistant(),
])
```

Violation sequence: (1) stop LLM generation, (2) flush the TTS queue and cut audio, (3) speak the scripted
fallback line, (4) keep the blocked text out of the assistant context, (5) log the decision.

**Must verify in Pipecat source before building** (not yet checked):
- Which frames the **assistant context aggregator** builds history from: LLM text or spoken (TTS) text.
  If LLM text, blocked sentences leak into history unless suppressed. Pipecat 1.14 notes the context now keeps
  the LLM's original text even with `text_filters`, which suggests it's LLM text.
- How to **cancel LLM generation mid-reply** from inside a processor (interruption frames / interruption
  task frames) without the rest of the pipeline treating it as the *user* barging in.
- How the guard should coexist with `LLMTextProcessor` and `text_transforms` (1.14+): before them, after
  them, or implemented as one.
- Which Pipecat version to target. `text_transforms` needs 1.14+; a plain `FrameProcessor` works on 1.12.

Bugs in an earlier code sketch (from another chat) to avoid:
- It didn't reset its buffer on `InterruptionFrame`.
- It ignored tool-call / function-call frames (they must pass through; tool arguments may need their own checks).
- It pushed plain `TextFrame`s. In 1.x, LLM output is `LLMTextFrame`, and `LLMTextProcessor` emits
  `AggregatedTextFrame`; check what the TTS service expects.
- It must forward every frame type it doesn't handle (a common custom-processor bug).

Useful Pipecat facts (checked in 1.12.0 source):
- `pipecat.classifiers.base_classifier`: `BaseClassifier`, `YesNoQuestion(instructions, yes, no)`,
  `ChoiceQuestion`, `ScoreQuestion`, `ClassifierError`; results have `is_yes` / `probability`; metrics via
  `on_metrics`. Classifiers are plain objects, not pipeline processors.
- `pipecat.processors.aggregators.llm_text_processor.LLMTextProcessor` turns `LLMTextFrame`s into
  `AggregatedTextFrame`s through a `BaseTextAggregator` (`SimpleTextAggregator` = by sentence,
  `PatternPairAggregator`, `SkipTagsAggregator`). It resets on `InterruptionFrame` and flushes on
  `LLMFullResponseEndFrame`.
- Frames can carry `skip_tts` (text reaches context and observers but isn't spoken).
- `PipelineTask` / `PipelineRunner` are deprecated since 1.3.0; use `PipelineWorker` / `WorkerRunner`.
- 1.15 added `PipecatToneTagger` and `ExpressionMetadataFrame`, other processors between LLM and TTS. Check
  ordering with them.

## 6. Fast rails detail
- **Terms**: case-folded, word-boundary matching across Roman and Devanagari; optional variant lists /
  transliteration-aware matching; per-term action (block / redact / flag). Typical uses: competitor names,
  banned phrases.
- **Indian PII**: Aadhaar (12 digits, Verhoeff checksum, spaced or hyphenated), PAN (`AAAAA9999A`), +91 / 10-digit
  mobile, UPI ID (`name@bank`), card numbers (Luhn), email. Presidio as an optional extra (needs a spaCy model,
  large; not installed on the dev machine).

## 7. Benchmark
- Dataset: JSONL of agent replies `{text, lang, labels: [rule ids violated], notes}`. Start with about 150 English,
  Hinglish (Roman) and Hindi (Devanagari) replies: safe ones, clear violations, and adversarial near-misses
  ("dose ke baare mein doctor se poochiye" is safe; "dose double kar lijiye" is not).
- Labels written by an AI model need a human (native speaker) review before the numbers are trusted.
- Report per judge: precision, recall, F1 per rule; p50/p95 latency; cost per 1k checks.
- API judges spend real credits on every run.

## 8. Dev environment (the user's machine)
- Windows 11, Python 3.12.10 (pyenv-win), PowerShell + Git Bash.
- **No GPU**: torch 2.14.0+cpu, transformers 5.17.0 installed. Only small local models are practical.
- Presidio not installed.
- **Git: the user commits and pushes themselves** (SSH signing needs a local passphrase). The agent may
  stage, branch, diff, but must never run `git commit` or `git push`.

## 9. First consumer: VoiceMitra (`D:\dev\Projects\VoiceMitra`)
A voice medicine-adherence agent (Hindi/Hinglish/English) that calls patients. Pipecat 1.12.0 handles audio
only (LiveKit room, Sarvam STT/TTS). **The LLM loop is its own code, not Pipecat's**, and it checks the
**whole reply before speaking** (deliberate: a blocked reply is silently regenerated, so the patient never
hears it). It will use this library's `Guard.check()` in whole-reply mode first. Streaming comes later, and
only if latency metrics justify it.

Its current guardrail interface (`core/agent/guardrails.py`): text checks are
`check(text, context) -> text` and raise `GuardrailViolation(message, fallback_text=None)` to block.
Only the *content* rules move to this library; dialogue-state rules (fixed hand-off line after escalation,
"no question while wrapping up", tool-call argument checks, forced escalation) stay in VoiceMitra.

Content rules to port (current substring lists, a starting point for the term rail and the benchmark):
```python
DOSAGE_CHANGE_OR_DIAGNOSIS_PHRASES = [
    "dosage badha", "dosage kam", "dawai band kar do", "dawai badal do", "dose double",
    "increase the dose", "decrease the dose", "stop taking", "double the dose",
    "aapko ho sakta hai", "you might have", "this is likely",
]
ADVICE_OR_REASSURANCE_PHRASES = [
    "aaram kar", "rest kar", "take rest", "get some rest", "get rest",
    "paani pi", "drink water", "drink plenty", "you should", "i suggest", "i recommend",
    "aap ko chahiye", "aapko chahiye", "meri salah", "meri rai",
    "sab theek ho jayega", "sab theek ho jaega", "theek ho jayega", "everything will be fine",
    "will be fine", "don't worry", "do not worry", "chinta mat", "chinta ki koi baat nahi",
    "ghabraiye mat", "ghabrao mat",
]
```
Policies, as judge questions: no dosage-change advice; no diagnosis; no medical advice or reassurance;
(later) no other patient's data, no PII read aloud.

Its LLM clients (OpenAI-compatible), useful for the `openai_judge` adapter and benchmark:
- Sarvam: base URL `https://api.sarvam.ai/v1`, model `sarvam-105b-conversations`, env `SARVAM_API_KEY`.
  Quirks: once tool calls are in history, `tools` must stay non-empty; `tool_choice="none"` returned empty output.
- Gemini: base URL `https://generativelanguage.googleapis.com/v1beta/openai/`, model `gemini-3.5-flash-lite`,
  env `GEMINI_API_KEY` (about 500 free requests/day).
- Agent replies are short (about 15 words), Hinglish in Roman script, feminine voice ("kar rahi hoon").

## 10. Suggested build order
1. Core types, chunker, `Guard.check()` (whole reply), decision log, shadow mode; unit tests.
2. Fast rails: terms (Roman + Devanagari, variants) and Indian PII.
3. Judge layer: the `Decider` protocol, the `openai_judge` adapter (Gemini/Sarvam), and the
   `pipecat_classifier` adapter (Jev via `JevClassifier`, called over the API, not run locally).
4. Benchmark: labelled set + runner; compare rules only / Jev / Gemini / Sarvam / Laya.
   **Deliverable: a recommendation for which judge VoiceMitra should use, backed by the numbers.**
5. VoiceMitra integration in whole-reply mode (its content rules replaced by this library).
6. `StreamGuard` + Pipecat `OutputGuardProcessor` (after the open Pipecat questions in section 5 are checked).
7. Optional adapters: local HF models, Guardrails AI validators, Presidio.

## 11. Open decisions
- TypeSafe API key for Jev (the user may need to request early access).
- Target Pipecat version for the adapter (1.12 plain FrameProcessor vs 1.14+ text_transforms).
- Whether Laya / Qwen3Guard-0.6B are fast enough on CPU to be worth shipping as defaults.
- Repo license and whether to publish to PyPI.

## Sources
- Pipecat Jev docs: https://docs.pipecat.ai/api-reference/server/classifiers/jev
- Pipecat CHANGELOG: https://github.com/pipecat-ai/pipecat/blob/main/CHANGELOG.md
- Pipecat LLMTextProcessor: https://reference-server.pipecat.ai/en/latest/api/pipecat.processors.aggregators.llm_text_processor.html
- NeMo Guardrails output rail streaming: https://docs.nvidia.com/nemo/guardrails/latest/configure-guardrails/yaml-schema/streaming/output-rail-streaming
- pipecat-firewall: https://github.com/rushjais/pipecat-firewall
- GovTech SG on Jev/Kev for moderation: https://blog.ai.gov.sg/decision-models-for-guardrails-exploring-jev-and-kev-for-moderation/
- Red Hat decision-model benchmark: https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails
- Laya: https://huggingface.co/convaiinnovations/laya, https://github.com/NandhaKishorM/laya
- Kev: https://github.com/jaredpalmer/kev
- OpenDecider vs Laya/Jev: https://huggingface.co/blog/manjunathshiva/opendecider-beats-laya-and-jev
- Maya (unverified): https://medium.com/@visrow/llm-guards-vs-maya-a-lightweight-435m-yes-no-safety-gate-for-ai-agents-7e9fd5f23142
- Guard model comparisons: https://futureagi.com/blog/ultimate-guide-llm-guardrails-2026/, https://www.arxiv.org/pdf/2511.22047
