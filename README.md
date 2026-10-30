# pipecat-guardrails

Output guardrails for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents. It checks what
your bot is about to say, and stops it before it is spoken.

> **Status: early development.** Version 0.0.1 contains the package structure only, with no
> guardrails yet. Follow the repo for the first usable release (0.1.0).

## The problem

Pipecat has no built-in output guardrail, so every team writes its own processor between the LLM
and the TTS. These homemade processors tend to break on user interruptions, keep speaking after a
violation, let blocked text into the LLM history, and miss paraphrases and Hindi/Hinglish text.

This package will provide one processor you add to your pipeline:

```python
pipeline = Pipeline([
    transport.input(), stt, context_aggregator.user(), llm,
    OutputGuardProcessor(guard),   # checks each sentence before it is spoken
    tts, transport.output(), context_aggregator.assistant(),
])
```

There will also be a plain `await guard.check(text)` for apps that run their own LLM loop.

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
