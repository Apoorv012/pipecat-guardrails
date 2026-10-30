# pipecat-guardrails

Output guardrails for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents. It checks what
your bot is about to say, and stops it before it is spoken.

> **Status: early development.** Built: the rule layer (banned terms, regex, personal data,
> India pack), `guard.check()`, and the Pipecat processor. Next: the classifier and LLM-judge
> layers. See [PLAN.md](https://github.com/Apoorv012/pipecat-guardrails/blob/main/PLAN.md).

## In a Pipecat pipeline

Put `OutputGuardProcessor` between the LLM and the TTS. That is the only change to your bot.

```python
from pipecat_guardrails import Guard
from pipecat_guardrails.pipecat import OutputGuardProcessor
from pipecat_guardrails.rules import PII, BannedTerms, pii

guard = Guard(
    rules=[
        BannedTerms(["acme corp"], id="no-competitors"),
        PII(pii.email(), pii.phone(), id="pii"),
    ],
    fallback="Sorry, I can only help with our own products.",
)

pipeline = Pipeline([
    transport.input(), stt, context_aggregator.user(), llm,
    OutputGuardProcessor(guard),   # checks each sentence before it is spoken
    tts, transport.output(), context_aggregator.assistant(),
])
```

- Each sentence is checked before the TTS gets it, so nothing unchecked is spoken.
- On a block, the rest of the reply is dropped and the fallback is spoken instead. Blocked text
  never reaches the TTS, so it never enters the conversation history either.
- User interruptions, function calls and lines your own code speaks (`TTSSpeakFrame`) are
  handled and pass through in order.
- `OutputGuardProcessor(guard, strategy="full_reply")` checks the whole reply before speaking
  any of it: safer for rules that need the full reply, at the cost of latency.
- Events: `@processor.event_handler("on_violation")` and `"on_check"`.

> **Using RTVI (a Pipecat client SDK)?** By default Pipecat sends every LLM token to the client
> app *before* the guard checks it. Pass
> `rtvi_observer_params=guarded_rtvi_observer_params()` (from `pipecat_guardrails.pipecat`) to
> `PipelineWorker` to turn that off.

Try it without API keys: `uv run python examples/pipeline_demo.py`. A full voice bot is in
[`examples/pipecat_bot.py`](examples/pipecat_bot.py).

## Without Pipecat's LLM loop

If your app runs its own LLM loop, check whole replies directly. Everything is opt-in: the
guard only runs the rules you give it.

```python
from pipecat_guardrails import Guard
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, pii

guard = Guard(
    rules=[
        BannedTerms(["acme corp", "globex"], id="no-competitors"),
        BannedTerms(
            ["dawai band kar do", "chinta mat"],
            id="no-medical-advice",
            word_pattern=india.hinglish_word_pattern,   # also matches "davai", "cheenta", ...
            variants={"dawai": ["दवाई"]},
        ),
        PII(pii.email(), pii.card(), india.aadhaar(), india.phone(), id="pii"),  # redacts
    ],
    fallback="Sorry, I can't help with that.",
)

verdict = await guard.check("Aap davaai band kar do.")
verdict.blocked     # True
verdict.text        # "Sorry, I can't help with that."
verdict.rule_ids    # ("no-medical-advice",)
```

- **Actions:** each rule can `block` (speak the fallback instead), `redact` (replace the match,
  e.g. `[phone number]`), or `flag` (report only).
- **Opt out:** pass `enabled=False` to a rule, or call `guard.disable("pii")` at runtime.
- **Shadow mode:** `Guard(..., mode="shadow")` reports what it would do without changing
  anything, so you can try new rules on live traffic.
- **Errors:** a rule that crashes fails closed (blocks) by default. Use `on_error="allow"` to
  fail open instead.
- **Custom rules:** any function works:
  ```python
  @rule(id="no-urls")
  def no_urls(text: str) -> bool:
      return "http" in text
  ```

Runnable examples, the same health bot in two languages:
[`examples/whole_reply_english.py`](examples/whole_reply_english.py) and
[`examples/whole_reply_hinglish.py`](examples/whole_reply_hinglish.py).

## Why

Pipecat has no built-in output guardrail, so every team writes its own processor between the LLM
and the TTS. These homemade processors tend to break on user interruptions, keep speaking after a
violation, let blocked text into the LLM history, and miss paraphrases and Hindi/Hinglish text.

See [BEFORE_AFTER.md](https://github.com/Apoorv012/pipecat-guardrails/blob/main/BEFORE_AFTER.md)
for a side-by-side comparison and
[PLAN.md](https://github.com/Apoorv012/pipecat-guardrails/blob/main/PLAN.md) for the design and
roadmap.

## Install

```bash
pip install pipecat-guardrails          # includes Pipecat
pip install "pipecat-guardrails[jev]"   # also installs the TypeSafe Jev classifier client
```

Requires Python 3.11+ and Pipecat 1.12 or later (installed automatically).

## How checking works

Each sentence goes through up to three layers, and stops at the first one that is sure:

1. **Banned words, PII, regex**: local and instant.
2. **Classifier model** (optional), such as TypeSafe Jev: a fast yes/no model with an API key.
   A confident answer decides; an unsure one moves to layer 3.
3. **LLM judge** (optional): any OpenAI-compatible LLM gives the final answer.

Layers 2 and 3 use Pipecat's own classifiers (`pipecat.classifiers`), so any classifier Pipecat
supports works here.

## Development

```bash
uv sync                      # create .venv with dev tools
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pyright               # type check
uv run pre-commit install    # run the checks on every commit
```

## License

BSD-2-Clause, the same as Pipecat.
