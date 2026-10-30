# Output guardrails in Pipecat: before and after

The same voice bot three ways: without a guardrail, with a hand-written guardrail, and with
`pipecat-guardrails`.

The Pipecat code uses the Pipecat 1.15 class names. The "after" code shows the planned API from
`PLAN.md`; that package hasn't been built yet.

---

## The bot

A pharmacy assistant you talk to through your mic and speakers. It turns your speech into text,
sends it to the LLM, and speaks the reply.

```python
import asyncio, os
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-mini")
    tts = CartesiaTTSService(api_key=os.environ["CARTESIA_API_KEY"], voice_id="<voice-id>")

    context = LLMContext([{"role": "system", "content": "You are a pharmacy assistant. Keep replies short."}])
    agg = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))

    pipeline = Pipeline([
        transport.input(), stt, agg.user(),   # user speaks -> text -> added to history
        llm,                                  # LLM streams the reply
        tts, transport.output(),              # text -> audio -> speakers
        agg.assistant(),                      # what was spoken -> added to history
    ])
    await WorkerRunner().run(PipelineWorker(pipeline))

asyncio.run(main())
```

Nothing stops this bot from saying something it shouldn't.

---

## Before: a hand-written guardrail

Most people start with the system prompt:

```
Never mention competitors. Never tell anyone to stop or change their medication.
```

That asks the LLM to behave. It doesn't enforce anything, and it fails now and then.

Teams that go further write their own processor and put it between `llm` and `tts`:

```python
from pipecat.frames.frames import LLMTextFrame, LLMFullResponseEndFrame, TextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor

BANNED = ["acme pharmacy", "stop taking", "double the dose"]

class MyGuard(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.buffer = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            self.buffer += frame.text
            if self.buffer.rstrip().endswith((".", "?", "!")):
                sentence, self.buffer = self.buffer, ""
                if any(b in sentence.lower() for b in BANNED):
                    await self.push_frame(TTSSpeakFrame("Sorry, I can't help with that."))
                else:
                    await self.push_frame(TextFrame(sentence))
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.buffer:
                await self.push_frame(TextFrame(self.buffer))
                self.buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

# pipeline: ..., llm, MyGuard(), tts, ...
```

It works in a demo. On real calls:

| Problem | What happens |
|---|---|
| No reset when the user interrupts | Half a sentence from the old reply is glued onto the next one |
| The LLM keeps going after a block | "Sorry, I can't help with that." followed by the rest of the bad reply |
| Substring matching | "don't stop taking it" gets blocked; Hindi `दवाई` never matches |
| Crude sentence split | "Take 2.5 mg" is cut at "2."; "Dr. Rao" is cut at "Dr." |
| The last fragment isn't checked | Bad text with no full stop at the end is spoken anyway |
| Raw LLM text still reaches the client app | With Pipecat's defaults, blocked text can appear in the on-screen transcript (not yet tested) |
| Exact phrases only | Paraphrases like "you can skip your tablets" get through |
| No logs or tests | No way to know how often it fires or whether it blocks good replies |

Every team writes this class again and finds these bugs one at a time.

---

## After: with `pipecat-guardrails`

```python
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from pipecat.services.openai.llm import OpenAILLMService
from pipecat_guardrails import Guard
from pipecat_guardrails.rails import BannedTerms
from pipecat_guardrails.pipecat import OutputGuardProcessor

guard = Guard(
    # Layer 1: local and instant
    rules=[BannedTerms(["acme pharmacy"], id="no-competitors")],
    # Rules for layers 2 and 3, written once as yes/no questions
    policies={
        "no-med-advice": "Does this reply tell the user to stop, skip or change their medication?",
    },
    # Layer 2 (optional): fast classifier; sure answers decide, unsure ones go to layer 3
    classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
    # Layer 3 (optional): any OpenAI-compatible LLM gives the final answer
    judge=LLMClassifier(llm=OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-nano")),
    fallback="Sorry, I can't advise on that. Please check with your doctor.",
    mode="enforce",   # "shadow" logs violations without blocking
)

output_guard = OutputGuardProcessor(guard)

@output_guard.event_handler("on_violation")
async def on_violation(processor, verdict):
    logger.warning(f"blocked {[v.rail_id for v in verdict.violations]}")

pipeline = Pipeline([
    transport.input(), stt, agg.user(), llm,
    output_guard,                               # the only change to the bot
    tts, transport.output(), agg.assistant(),
])
```

What the package takes care of:

- It resets when the user interrupts.
- On a block, it drops the rest of the reply and speaks the fallback once.
- Blocked text is never spoken, so it never enters the LLM's history.
- It splits on real sentence boundaries (`2.5 mg`, `Dr.` and the Hindi `।` are handled).
- It matches whole words, in English and Devanagari.
- The classifier and the LLM judge catch paraphrases. The classifier settles the clear cases
  quickly, so the slower LLM only sees the unclear ones. Each policy has a timeout and a fail-open
  or fail-closed setting.
- Both use Pipecat's own classifiers, so any classifier Pipecat supports works.
- While one sentence is spoken, the next is checked, so only the first sentence waits for the judge.
- It ships a helper that stops raw LLM text from reaching the client app.
- Events, latency metrics and a shadow mode show how the rules behave before you enforce them.

Apps that run their own LLM loop, such as VoiceMitra, use the same guard on the whole reply:

```python
verdict = await guard.check(reply_text)
if verdict.blocked:
    reply_text = verdict.fallback_text
```

---

## Summary

| | Before | After |
|---|---|---|
| Code you write | A custom processor, about 30 lines, with the bugs above | Rules as config, plus one line in the pipeline |
| Interruptions | Breaks | Handled |
| After a violation | Bad reply keeps playing | Rest of the reply dropped, fallback spoken |
| LLM history | Depends on how you wrote it | Blocked text never enters it |
| Paraphrases | Missed | LLM judge |
| Hindi / Hinglish | Missed | Supported |
| Knowing if it works | Guesswork | Logs, shadow mode, eval command |
