# pipecat-guardrails

Output guardrails for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents. It checks what
your bot is about to say, and stops it before it is spoken.

> **Status: early development.** Built: the rule layer (banned terms, regex, personal data,
> India pack), policies answered by a classifier and an LLM judge, `guard.check()`, the Pipecat
> processor, and the `eval` command. Next: hardening and docs for a beta. See
> [PLAN.md](https://github.com/Apoorv012/pipecat-guardrails/blob/main/PLAN.md).

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

## Policies: checks that need understanding

Word lists catch exact words. They miss paraphrases ("you don't need the insulin anymore") and
block safe sentences that contain the words ("don't stop taking it without asking your
doctor"). For those, write a **policy**: a yes/no question about the reply, where "yes" means
it breaks the policy.

```python
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from pipecat.services.openai.llm import OpenAILLMService
from pipecat_guardrails import Guard, Policy

guard = Guard(
    rules=[...],                       # layer 1, as above
    policies={
        "no-med-advice": Policy(
            "Does this reply tell the user to stop, skip or change a medication?",
            no="Telling them to ask their doctor first, or to keep taking it as prescribed.",
        ),
        "no-diagnosis": "Does this reply tell the user what illness they have?",
    },
    classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),        # layer 2
    judge=LLMClassifier(llm=OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"],
                        settings=OpenAILLMService.Settings(model="gpt-4.1-nano"))),  # layer 3
    fallback="That's a question for your doctor.",
)
```

Each sentence goes through up to three layers, and stops at the first one that is sure:

1. **Rules**: banned terms, PII, regex. Local and instant. A block here skips the rest.
2. **Classifier** (optional), such as TypeSafe Jev: answers every policy in one call. At or
   above `block_above` (default 0.8) the policy fires; at or below `allow_below` (0.2) it
   passes; in between, the judge decides. With no judge, the classifier decides at 0.5.
3. **Judge** (optional): any OpenAI-compatible LLM gives the final yes or no for the policies
   that are still open.

- Use a classifier, a judge, or both. Both are Pipecat classifiers (`pipecat.classifiers`), so
  any classifier Pipecat supports works, and you can subclass `BaseClassifier` for your own.
- The classifier and judge see the text after redaction, so personal data isn't sent to them.
  When you call `guard.check()` yourself, pass `CheckContext(history=...)` to give them the
  last few turns (`history_turns=4`). Inside a pipeline they don't see earlier turns yet, and
  with `strategy="sentence"` they see one sentence at a time.
- Timeouts: `classifier_timeout=2.0`, `judge_timeout=5.0` seconds. A failed or slow call
  follows `on_error` (`"block"` by default); `Policy(on_error="allow")` fails open for one
  policy. A classifier failure goes to the judge when there is one.
- In the processor, sentences are checked concurrently and released in order, so usually only
  the first sentence of a reply waits for a full round trip. `max_parallel_checks=5` caps the
  calls in flight; after a block, sentences still waiting are never sent.
- `Policy(action="flag")` reports without blocking. Shadow mode works for policies too.

## Measure it on your own data

Before you trust a rule or a policy, score it on replies you've labelled. One JSON object per
line; `labels` lists the rules or policies that should fire (empty means the reply is fine):

```json
{"text": "You can stop taking it for a few days.", "labels": ["no-med-advice"]}
{"text": "Please don't stop taking it without asking your doctor.", "labels": []}
```

```bash
pipecat-guardrails eval data.jsonl --guard app/guards.py:guard
```

You get precision, recall and F1 per label, the latency of each check (the delay before the
first sentence is spoken), the error rate, and `--verbose` lists every mistake. Pass several
`--guard`s to compare setups side by side, `--repeat 3` for LLM judges that vary between runs,
`--json report.json` to save the report, and `--diff report.json` to fail CI when F1 drops. A
rule `no-med-advice:terms` counts for the label `no-med-advice`, so rules and policies are
scored on the same question.

A sample dataset (a health bot in English and Hinglish) and four guards, from rules only to all
three layers, are in [`examples/eval/`](examples/eval/). On it, adding the Jev classifier to the
rules raised F1 from 0.70 to 0.96 at about 310 ms per check ([results](docs/eval/README.md)):

```bash
uv run pipecat-guardrails eval examples/eval/sample.jsonl --guard examples/eval/guards.py:rules_only
```

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
