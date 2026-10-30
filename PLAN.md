# pipecat-guardrails: plan

Status: draft, 2026-10-30. Pipecat facts below were checked against the `pipecat-ai` 1.15.0 wheel
and the installed 1.12.0 source, unless marked *to verify*.

---

## 1. The problem

Teams building voice agents on Pipecat need to stop the bot from **saying** certain things: competitor
names, PII, medical or financial advice, off-brand promises. Pipecat 1.15 has no output guardrail, so
each team writes its own `FrameProcessor` between the LLM and the TTS. These homemade processors tend
to get the same things wrong:

- they don't reset on user interruptions, so text from an old reply leaks into the next one;
- they swallow frames they don't handle (function calls, control frames), which breaks the pipeline;
- they buffer too much, adding latency, or too little, checking half-sentences;
- they can't decide what to do after a violation: is the earlier sentence already playing, does the
  blocked text end up in the LLM history, what does the user hear instead;
- they have no shadow mode, no logs and no way to measure false positives.

**Goal:** `pip install pipecat-guardrails`, put one processor between the LLM and the TTS, declare rules,
and get correct, low-latency, observable enforcement that works across Pipecat releases.

```python
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.services.openai.llm import OpenAILLMService
from pipecat_guardrails import Guard
from pipecat_guardrails.rules import BannedTerms, PII
from pipecat_guardrails.pipecat import OutputGuardProcessor

guard = Guard(
    rules=[
        BannedTerms(["acme corp", "globex"], id="no-competitors"),
        PII(entities=["email", "phone", "card"], action="redact"),
    ],
    policies={"no-dosage": "Does this reply recommend changing a medication dose?"},
    judge=LLMClassifier(llm=OpenAILLMService(model="gpt-4.1-nano")),
    fallback="Sorry, I can't help with that. Is there anything else I can do?",
)

pipeline = Pipeline([
    transport.input(), stt, context_aggregator.user(), llm,
    OutputGuardProcessor(guard),          # <- the only change to the pipeline
    tts, transport.output(), context_aggregator.assistant(),
])
```

The same `Guard` also works without Pipecat: `verdict = await guard.check(text)`.

### Scope

| In scope (v0.x) | Out of scope |
|---|---|
| Checking the bot's **output** text before it is spoken | Checking user input (prompt injection, etc.); `pipecat-firewall` covers that |
| Built-in rules (terms, regex, PII, custom functions), plus the classifier → LLM-judge cascade using Pipecat's classifiers | Training or hosting safety models; writing our own API clients for judges |
| Pipecat processor with correct voice behaviour | Cost or budget limits (`floe-guard` covers those) |
| Shadow mode, decision events, latency metrics | A server or proxy mode |
| An eval command to measure your own rules on your own data | A built-in "best judge" ranking |

### What I left out of the context notes, and why

The notes mixed the library with one app (VoiceMitra) and a research project. They are separate things.

| From the notes | Decision |
|---|---|
| Comparing Jev, Laya, Kev, Maya, Sarvam and Gemini as judges | **Not part of the library.** The library ships a judge interface and an eval command. Picking a judge for VoiceMitra is a study that *uses* the library, and its results go in a blog post or `docs/`, not into the code. |
| VoiceMitra medical phrase lists | **An example**, not core. Ship it under `examples/` as a sample policy. |
| India as the headline feature | **A rule pack** (`pipecat_guardrails.packs.india`: Aadhaar, PAN, UPI, +91, Hinglish/Devanagari matching). It's a strong differentiator, but the core stays language-neutral. It's in v0.1 because the first user needs it. |
| Adapters for Guardrails AI, Hugging Face local models and Presidio | **Later** (milestone 5). Each pulls in heavy dependencies and none blocks v0.1. |
| "Background judge that cuts audio after the fact" | **Experimental, later.** It has a context-history problem (section 2, finding 3) that needs its own design. |
| Long lists of model versions, prices and free tiers | Dropped. These change monthly and don't belong in a plan. |

---

## 2. What the Pipecat source says (and how it changes the design)

1. **The LLM history is built from text that was actually spoken, not raw LLM text.** The TTS sets
   `append_to_context = False` on the text it receives (`services/tts_service.py`, `_announce_text`) and
   emits `TTSTextFrame`s as audio plays. Those are what the assistant aggregator records
   (`llm_response_universal.py`, `_handle_text`). Every `TextFrame` has an `append_to_context` flag.
   **Consequence:** text the guard blocks before the TTS never reaches the history, with no extra work.
   This is the main reason to make blocking before the TTS the default.

2. **Any processor can interrupt.** `FrameProcessor.broadcast_interruption()` (present in 1.12 and 1.15)
   sends an `InterruptionFrame` both upstream, which cancels the LLM's in-flight generation, and
   downstream, which flushes the TTS and the output audio. **Consequence:** "stop the bot now" is one
   call. Side effect: the aggregator marks the turn `interrupted=True`, which shows up in analytics
   like a user barge-in. Document this, and add a flag to the `on_violation` event so apps can tell
   the two apart.

3. **On an interruption, the aggregator commits whatever was spoken so far**
   (`_handle_interruptions` → `_trigger_assistant_turn_stopped(interrupted=True)`). **Consequence:**
   if a violation is found *after* audio has started (the "background judge" idea), the part of the bad
   sentence that was already spoken goes into the history. Fixing that means repairing the context
   afterwards, for example with `LLMMessagesTransformFrame` (in 1.15; *to verify* for 1.12). That's why
   this strategy is deferred and marked experimental.

4. **The TTS speaks `AggregatedTextFrame`s as they arrive** (`_handle_aggregated_text`) and only re-chunks
   plain `TextFrame`s. **Consequence:** the processor emits one `AggregatedTextFrame(text, TextType.SENTENCE)`
   per approved sentence, so the TTS speaks exactly the units that were checked. The field was renamed
   from `aggregated_by` to `text_type` in 1.13. Passing it positionally works on both versions.

5. **The raw LLM text can reach the client UI before the guard sees it.** `RTVIObserver` forwards
   `LLMTextFrame`s to the client (`bot_llm_enabled` defaults to `True`; `bot-llm-text` is deprecated but
   still sent). Observers see frames at the LLM, upstream of the guard. **Consequence:** blocked text can
   show up in a client transcript. Ship a helper and a README warning. The fix is
   `RTVIObserverParams(bot_llm_enabled=False)` or adding the LLM to `ignored_sources`, and relying on the
   `bot-output` messages, which come from the output transport after the guard. *To verify* with a test.

6. **There's nothing to reuse for this in Pipecat.** `StatelessTextTransformer` and `text_transforms`
   can only rewrite text. They can't block, interrupt or replace a reply. A plain `FrameProcessor`
   is the right base, and it works from 1.12 on.

7. **Test tools exist.** `pipecat.tests.utils.run_test(processor, frames_to_send=..., expected_down_frames=...)`
   handles frame-level tests, and `pipecat eval run` (1.15 `pipecat/evals`) handles end-to-end tests.
   The community-integration guide asks for the latter.

---

## 3. Architecture

One package, with Pipecat as a required dependency. Pipecat already provides the classifier and
LLM-judge backends (`pipecat.classifiers`: `JevClassifier`, `LLMClassifier`,
`OpenAIDecisionsClassifier`, `ClefClassifier`), the OpenAI SDK, and the processor base class. So
this package writes no API clients. It adds the local rules, the cascade that ties the layers
together, and the voice enforcement.

`Guard.check(text)` doesn't need a running pipeline, so apps with their own LLM loop (VoiceMitra)
call it directly.

### The three layers

```
sentence
  → 1. rules        banned terms, PII, regex, custom    match?               → BLOCK / REDACT
  → 2. classifier   e.g. JevClassifier (optional)       p ≥ block_above      → BLOCK
                                                        p ≤ allow_below      → ALLOW
                                                        in between           → next layer
  → 3. judge        LLMClassifier over any OpenAI-      final yes/no         → BLOCK / ALLOW
                    compatible LLM (optional)
```

- Policies are written once as yes/no questions (`{"no-dosage": "Does this reply ...?"}`) and
  passed to both layers as Pipecat `YesNoQuestion`s. One `yes_no()` call per sentence answers all
  of them.
- Any layer is optional. With no classifier, sentences go straight from rules to the judge. With no
  judge, the classifier decides alone at `p ≥ 0.5`.
- `LLMClassifier` probabilities aren't calibrated (Pipecat's docstring says so), so layer 3 uses
  `is_yes` as a final answer, not a threshold.
- Any other model (Maya, a local Laya, an in-house API) plugs in by subclassing Pipecat's
  `BaseClassifier`.
- Default thresholds: `allow_below=0.2`, `block_above=0.8`, both set per policy. Tune them with the
  eval command.

```
src/pipecat_guardrails/
  __init__.py            Guard, Verdict, Violation, Action, re-exports
  types.py               Verdict, Violation, Action(ALLOW|REDACT|BLOCK|FLAG), CheckContext
  guard.py               Guard: rules → classifier → judge cascade, mode and fail policy
  rules/
    base.py              Rule protocol + @rule decorator for plain functions
    terms.py             BannedTerms: Unicode-normalised, word-boundary, variant lists
    regex.py             Regex
    pii.py               PII: email, phone (E.164), card (Luhn), IP; pluggable recognisers
  policies.py            policy questions → pipecat YesNoQuestion; thresholds per policy
  packs/
    india.py             Aadhaar (Verhoeff), PAN, UPI, +91; Roman/Devanagari term variants
  text/
    chunker.py           streaming sentence splitter (. ? ! । ॥ \n), abbreviation/decimal safe
    normalize.py         NFKC, casefold, strip expression tags, transliteration hooks
  pipecat/
    processor.py         OutputGuardProcessor(FrameProcessor)
    rtvi.py              helper to stop raw LLM text reaching clients
  eval/
    cli.py               `pipecat-guardrails eval data.jsonl --guard mymodule:guard`
  py.typed
```

### Core contracts

```python
class Rule(ABC):                             # layer 1; local and synchronous-fast
    id: str; action: Action; enabled: bool   # every rule is opt-in and can be switched off
    def find(self, text: str, context: CheckContext) -> Iterable[Match]: ...

# Layers 2 and 3 are any pipecat.classifiers.base_classifier.BaseClassifier.

@dataclass(frozen=True)
class Verdict:
    action: Action                           # most severe action across rules
    text: str                                # original, or redacted when action is REDACT
    violations: tuple[Violation, ...]        # rail_id, reason, span, score, action
    latency_ms: dict[str, float]             # per rail
    shadow: bool                             # True if it would have acted but mode="shadow"
```

`Guard.check()` order:
1. Run the rules. Stop on BLOCK. Apply REDACT spans from the end of the string backwards so the
   offsets stay valid.
2. Classifier (if set): one `yes_no()` call for all policies, with a timeout. Policies above
   `block_above` block; those below `allow_below` pass; the rest go to the judge.
3. Judge (if set): one `yes_no()` call for the policies that are still undecided, with a timeout.
4. Merge: BLOCK > REDACT > FLAG > ALLOW.
5. On a timeout or `ClassifierError`, apply the policy's `on_error` (`"allow"` = fail open,
   `"block"` = fail closed, `"escalate"` = try the next layer). Record it as a violation of kind
   `error` so it shows up in metrics.
6. `mode="shadow"`: compute everything, return ALLOW with `shadow=True`, and emit the event.

Classifier metrics (`on_metrics` on `BaseClassifier`) are passed through to the guard's latency
report.

---

## 4. The Pipecat processor: behaviour spec

### Strategies

| Strategy | What happens | Added latency | Guarantee |
|---|---|---|---|
| `"sentence"` (default) | Buffer to a sentence boundary, check, release. Sentence N+1 is checked while N is being spoken | Rules: ~0.2 ms. Judge: one round trip on the first sentence only | Nothing unapproved is ever spoken |
| `"full_reply"` | Buffer the whole reply, check once, then release it sentence by sentence | Whole LLM generation + check | Nothing unapproved is spoken, and a block replaces the entire reply |
| `"optimistic"` *(experimental, later)* | Speak right away, judge in the background, cut audio on a violation | None | Weaker: a fragment may be heard and needs context repair (section 2, finding 3) |

### Frame handling

| Frame | Behaviour |
|---|---|
| `LLMFullResponseStartFrame` | Start a new reply state and forward the frame |
| `LLMTextFrame` | Feed the chunker; check and emit complete sentences as `AggregatedTextFrame` |
| `AggregatedTextFrame` (when `LLMTextProcessor` runs upstream) | Check it as one unit with no re-chunking, then forward it unchanged |
| `TextFrame` with `skip_tts=True` | Still checked (it can reach the context and the client UI); drop it if blocked |
| `LLMFullResponseEndFrame` | Flush the chunker remainder, check it, then forward the end frame |
| `InterruptionFrame` | Cancel pending checks, clear the buffer and reply state, forward the frame |
| `LLMThoughtTextFrame`, function-call frames, `TTSSpeakFrame`, all others | Forward untouched, in order. Scripted `TTSSpeakFrame`s are not checked by default (`check_tts_speak=False`) |
| Any upstream frame | Forward untouched |
| `EndFrame` / `CancelFrame` | Cancel background tasks and forward the frame |

Ordering rule (built in milestone 2): when a non-text frame arrives while text is held, the held
text is checked and passed on first, so frame order downstream matches the order the LLM produced.
In `full_reply`, non-text frames are held with the reply and released in their original positions;
on a block the fallback takes the place of the first text and function-call frames still pass.

### When a sentence is blocked (strategy `sentence`, as built)

1. Stop releasing text for this reply. Drop the rest of the reply's text until `LLMFullResponseEndFrame`.
2. Sentences approved earlier keep playing (they were checked).
3. The LLM is not interrupted: the rest of the reply is drained and discarded. Replies are short,
   and this avoids interruption side effects (decision 3).
4. Emit the fallback line as one more `AggregatedTextFrame` sentence inside the same reply. The
   history then reads "approved part + fallback" as one assistant message. `fallback=None` means
   silence; a callable `(verdict) -> str` picks per-rule or per-language lines.
5. The blocked text never reaches the TTS, so it never enters the history (section 2, finding 1).
6. Fire `on_violation(processor, verdict)` and `on_check`.

Deferred: `on_block="cut"` (flush approved sentences too) and `context_note=True` (tell the LLM its
reply was withheld). Add them if real use asks for them.

REDACT replaces the matched span (`"call me at [phone number]"`) and releases the sentence. It never
interrupts. FLAG releases unchanged and only fires the event.

### Coexisting with other processors
- Default position: directly before the TTS, so the guard checks the text that will be spoken.
- Expressive sound and pause units (1.15 `expressive`) pass through unchecked. Tags inside text
  are not stripped yet; rules aren't affected by them. Revisit for the LLM judge (milestone 3).
- Speech-to-speech (realtime) LLMs are out of scope: they produce audio directly, so there is no
  text to check before it is spoken.
- One integration test per known neighbour: `LLMTextProcessor`, `PipecatToneTagger`, a function-calling
  LLM, an interruption mid-check.

### Events and observability
- `on_violation`, `on_check` (every decision, for sampling) and `on_error`, using Pipecat's
  `event_handler` decorator.
- Per-rail latency through Pipecat metrics (`MetricsFrame` / processing metrics) when metrics are enabled.
- Structured log records (`rail_id`, action, latency, reply id, shadow) through `loguru` in the adapter
  and stdlib `logging` in the core.
- Never log the full blocked text by default (it may contain PII). `log_text=True` is opt-in.

---

## 5. Layers in detail (v0.x)

**Layer 1: rules (ours)**

| Rule | Notes |
|---|---|
| `BannedTerms` | NFKC and casefold. Word boundaries that work for Devanagari (no `\b`: Python's `re` treats combining marks as non-word characters, so `\b\w+\b` splits `दवाई` into `दव`, `ई`). Per-term variants. A single compiled alternation (or Aho-Corasick past about 1k terms) |
| `Regex` | Named patterns, per-pattern action |
| `PII` | Recognisers return spans with confidence. Checksums where they exist (Luhn, Verhoeff) to cut false positives. Default action is REDACT |
| `packs.india` | `AadhaarRecognizer`, `PANRecognizer`, `UPIRecognizer`, `IndianPhoneRecognizer`, a Hinglish variant generator (dawai/davai/दवाई) |
| `@rule` | The extension point for custom logic |

**Layers 2 and 3: Pipecat classifiers (we only wire them up)**

| Backend | Import | Layer | Needs |
|---|---|---|---|
| TypeSafe Jev | `pipecat.classifiers.typesafe.jev.classifier.JevClassifier` (1.13+; `pipecat.classifiers.jev` on 1.12) | 2 | `pipecat-guardrails[jev]`, TypeSafe API key |
| OpenAI Decisions | `pipecat.classifiers.openai.decisions.OpenAIDecisionsClassifier` (1.13+) | 2 | OpenAI key |
| Cloudflare Clef | `pipecat.classifiers.cloudflare.clef.ClefClassifier` (1.13+) | 2 | Cloudflare account |
| Any LLM | `pipecat.classifiers.llm.classifier.LLMClassifier(llm=OpenAILLMService(base_url=..., model=...))` | 3 | Key for that endpoint (Gemini, Sarvam, OpenAI, Groq, local vLLM) |
| Anything else | subclass `BaseClassifier` | 2 or 3 | — |

What the guard adds on top: a policy is a `YesNoQuestion(instructions, yes, no)`. The `yes`/`no`
fields hold the near-miss examples that matter for accuracy ("asking them to consult their doctor
is a no"). Each judge call can include recent turns (`CheckContext.history`) in its input.

---

## 6. Evaluation

`pipecat-guardrails eval data.jsonl --guard app.guards:guard [--rails id1,id2] [--repeat 3]`

- Input: one JSON object per line, `{"text": ..., "labels": ["rule-id", ...], "lang": "...", "notes": ...}`.
- Output: a table per rail with precision, recall, F1, p50/p95 latency and error/timeout rate.
  Optional `--json` output for CI and `--diff baseline.json` to catch regressions when rules change.
- Ships with a small sample dataset (English plus Hinglish) for the README and tests. Users bring
  their own for real numbers.
- The VoiceMitra judge comparison runs on this command as a separate piece of work.

---

## 7. Engineering standards

**Packaging**
- `uv` + `hatchling`, `src/` layout, `py.typed`. Python `>=3.11` (Pipecat 1.15's floor).
- Dependency: `pipecat-ai>=1.12,<2`, and nothing else.
- Extras: `jev` (passes through to `pipecat-ai[jev]`); later `local` (on-device models) and `presidio`.

**Quality gates (CI on every PR)**
- `ruff` (lint + format). `pyright` strict on `core/` and basic on adapters.
- `pytest` + `pytest-asyncio`. Coverage gate of 90% on the core.
- Chunker property tests (`hypothesis`): the chunks join back to the input; no split inside
  decimals, times or known abbreviations; it handles text streamed one character at a time.
- Frame-level tests with `pipecat.tests.utils.run_test` for every row of the frame table and every
  violation path.
- Latency test: rules under 1 ms p95 per sentence (measured 0.33 ms p95 with 38 terms plus all
  PII recognizers). CI uses a generous 5 ms budget to catch large regressions only.
- Sentence splitting is our own rule-based splitter, not Pipecat's `match_endofsentence`
  (sentencex) or spaCy. Pipecat's splits `"Take 2."` before `".5 mg"` arrives and cuts after
  `"Tab."`/`"Smt."`. spaCy needs whole text, is heavy, and has no trained Hindi pipeline.
- Matrix: Python 3.11–3.13 × Pipecat {1.12, latest}, plus a **nightly job against Pipecat `main`**,
  which warns early about breaking changes (the community guide says to expect them).
- End-to-end: the foundational example runs under `pipecat eval run` with a scripted user, mocked
  LLM/TTS and an asserted transcript.

**Release**
- SemVer with `0.x` until the API has had real use. `CHANGELOG.md` (Keep a Changelog).
- GitHub Actions publishing to PyPI through Trusted Publishing (no tokens). Tagged releases only.
- License **BSD-2-Clause**, matching Pipecat (the community guide asks for a permissive license).
- `README` (60-second quickstart, a "Tested with Pipecat vX" line, the RTVI warning), `CONTRIBUTING.md`,
  `SECURITY.md`, issue templates.
- Docs: start with the README plus `docs/` markdown. Add mkdocs-material once there's more than about
  4 pages.

**Getting it into Pipecat's docs** (per `COMMUNITY_INTEGRATIONS.md`)
- A single-file foundational example, a README with the tested Pipecat version, a permissive
  license and a changelog.
- PR to the Pipecat docs repo: a row in `supported-services.mdx`, a service page, a `docs.json` entry.
- A 30–60 s demo video showing a block, the fallback, and a user interruption during a check.
- Post in Discord `#community-integrations`. Keep the "last tested" version current, or the listing
  can be removed.

---

## 8. Milestones

| # | Deliverable | Done when |
|---|---|---|
| 0 | Repo scaffold: uv, CI, lint/type/test, license, empty package published as `0.0.1` to reserve the name | CI green; the name is reserved on PyPI |
| 1 | Core: types, `Guard.check`, chunker, `BannedTerms`, `Regex`, `PII`, `packs.india`, shadow mode, fail policy → **v0.1.0** | VoiceMitra swaps its content rules for `Guard.check()` in whole-reply mode |
| 2 | `OutputGuardProcessor` with the `sentence` and `full_reply` strategies, the full frame table, the violation sequence, the RTVI helper, the foundational example → **v0.2.0** | All frame tests pass on Pipecat 1.12 and latest; the example runs locally with real STT/TTS |
| 3 | Policies and the classifier → judge cascade (Pipecat classifiers), judge checks pipelined across sentences, the `eval` command → **v0.3.0** | Eval report on the sample set for rules only, rules+Jev, rules+LLM, and all three; first-sentence latency measured for each |
| 4 | Hardening: e2e `pipecat eval run`, nightly Pipecat-`main` job, docs, demo video, Pipecat docs PR → **v0.4.0 (beta)** | Listed in the Pipecat docs |
| 5 | Optional: `optimistic` strategy with context repair; Presidio, HF local model and Guardrails AI adapters; checks on tool-call arguments | Each one ships behind its own extra, with tests |
| 1.0 | API freeze after at least 2 external users and one release cycle without breaking changes | — |

Parallel track (not library work): the VoiceMitra judge comparison, using the milestone 3 `eval`
command and a labelled set that a native speaker has reviewed.

---

## 9. Decisions

Settled:
1. **Mid-reply block:** approved sentences finish, then the fallback plays. (Milestone 2.)
2. **Fallback in history:** yes. It is spoken by the TTS, so Pipecat records it like any reply.
3. **Stopping the LLM on block:** drain and discard the rest of the reply; no interruption.
4. **Minimum Pipecat version:** 1.12, tested in CI with the oldest allowed dependencies.
5. **Package name:** `pipecat-guardrails`, published (0.0.1, 0.1.0).

Open:
6. **Who maintains it:** the community listing requires tracking Pipecat releases. The nightly CI job
   makes that cheap, but someone has to act on its failures.
