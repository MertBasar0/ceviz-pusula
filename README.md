# Pusula

An optional model-routing plugin for the [Ceviz](https://github.com/MertBasar0/ceviz) helper. Ceviz
lets you send voice commands from Apple Watch to your own OpenClaw agent.

Without Pusula, every Ceviz command runs on the OpenClaw agent's own model, and OpenClaw's fallback
chain handles outages. Pusula keeps that default for ordinary turns. It steps in only when the
conversation shows that a request was missed, and then sends the retry to a stronger model you
already configured.

## What it does

- **Ordinary turns stay on the agent default.** Pusula returns no model, so Ceviz sends no
  `--model` and your fallback chain stays in play.
- **It escalates after misses, judged by a local decision model.** On every turn Pusula asks
  [APUS-OpenJev-v1 4B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF), served by
  Ollama on your machine, two questions about the new message and the last three requests: is the
  user pushing back on the previous answer, and is it asking again for something that did not
  finish?
  - **Level 1:** either answer is yes (probability 0.5 or more). The turn runs on the best
    configured "balanced" model (for example the newest Sonnet) at `medium` thinking.
  - **Level 2:** the previous job also ended blocked, unknown or failed, or the previous turn was
    itself a correction. The same model runs at `high` thinking.
  - A provider outage is not a miss, so it never escalates.
  - **If the decision model cannot answer** (Ollama is down, the model is missing, the answer is
    late or unreadable), the turn runs on the agent default. Nothing else guesses in its place.
  - Pusula used a phrase regex until v0.1.0. On the
    [escalation-detector set](experiments/escalation-detector/README.md) it caught 37% of misses
    at 53% precision; OpenJev 4B caught 99% at 92%.
- **It only uses models you configured.** Pusula reads `openclaw models list --agent <agent> --json`
  and considers only models tagged as configured, default or a fallback, from Anthropic, OpenAI,
  Google and xAI. A model that is merely available may sit behind a paid API key you never chose
  for Ceviz. The list is cached for 6 hours and refreshed in the background, so a routing
  decision never waits on the CLI once anything is known.
- **Light tier (optional, off by default).** The same decision model can mark fresh small-talk
  turns for a lighter model you name. Its question has not been measured on a labelled set yet,
  so it needs a confident yes (0.8 by default).

## Install and enable

Pusula needs Python 3.11 or newer and a Ceviz helper with router contract v1:
[`ceviz-helper-v2026.10.1-beta.2`](https://github.com/MertBasar0/ceviz/releases/tag/ceviz-helper-v2026.10.1-beta.2)
or newer. Escalation and the light tier also need [Ollama](https://ollama.com) with the
decision model (about 4.5 GB on disk, about 6.7 GB in memory with Pusula's 2048-token context).
Pusula keeps it loaded until Ollama stops, so that memory stays reserved:

```bash
# 0. The decision model
ollama pull hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0

# 1. Install into the helper's own environment
~/path/to/ceviz/.venv/bin/pip install git+https://github.com/MertBasar0/ceviz-pusula@v0.2.1

# 2. Enable it for the helper service
systemctl --user edit watch-ceviz-backend
#   [Service]
#   Environment=WATCH_CEVIZ_ROUTER=pusula
systemctl --user restart watch-ceviz-backend

# 3. Check
~/path/to/ceviz/deploy/doctor.sh   # Router plugin "pusula" is installed and enabled
```

Ceviz's [router plugin guide](https://github.com/MertBasar0/ceviz/blob/ceviz-helper-v2026.10.1-beta.2/docs/router-plugins.md)
describes what the helper guarantees around plugins:

- a failing or slow router never fails a command
- only plain selectors reach the command line
- a pinned command that fails before execution is retried once without the pin

To turn routing off, remove `WATCH_CEVIZ_ROUTER` and restart the service. To remove Pusula
entirely, run `pip uninstall ceviz-pusula` in the same environment.

## Configuration

On first use, Pusula writes `pusula.json` to the Ceviz state directory (by default
`~/.openclaw/ceviz-state`). The useful keys are:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch inside Pusula |
| `routing_mode` | `context_aware` | `context_aware`, `single_turn` (no escalation), or `disabled` |
| `enable_correction_escalation` | `true` | Escalate after corrections and repeated misses |
| `escalation_models` | `[]` | Explicit ladder, for example `["anthropic/claude-sonnet-5"]`. Empty means the best configured balanced model |
| `escalation_thinking` | `["medium", "high"]` | Thinking level per escalation step |
| `escalation_window_seconds` | `900` | How old a recent job may be and still count as evidence |
| `escalation_threshold` | `0.5` | Probability at which an escalation answer counts as yes |
| `escalation_history` | `3` | How many recent requests the decision model sees |
| `decision_model_url` | `http://127.0.0.1:11434` | Ollama server that runs the decision model |
| `decision_model` | `hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0` | Decision model for escalation and light turns |
| `decision_timeout_ms` | `2500` | Budget for all of a turn's questions. Pusula never lets it exceed the helper's routing timeout minus 200 ms (see below) |
| `light_instructions`, `light_threshold` | unset, `0.8` | Question and probability threshold for light turns |

v0.1 used `decision_endpoint` (a System One server such as Kev) for the light tier. v0.2 ignores
that key; both decisions go to `decision_model`.

### Timing

A turn asks the decision model two to four questions, one after another. A warm model answers each
in 0.2 to 0.5 s. The first turn after the model sat idle beside other GPU work measured 1.27 s and
0.98 s for its two questions, so the 1.2 s budget of v0.2.1 and earlier ran that turn on the agent
default. Two things now cover it:

- **A larger budget.** The default is 2500 ms. Helpers that state their routing timeout
  (`routing_timeout_ms`, Ceviz 3000 ms by default) cap it at that timeout minus 200 ms. Older
  helpers wait 1500 ms, so Pusula keeps their turns to 1300 ms.
- **A wake-up before the words are known.** Helpers that support the optional `prepare()` hook call
  it when a command is on its way, before speech recognition. If the decision model has not
  answered in the last 30 s, Pusula has it produce one token in the background (about 0.15 s when
  loaded). This matters most when Ollama no longer holds the model, for example after you freed
  the GPU for something else: loading took 2.8 to 4.6 s here, more than any routing budget, but with the
  wake-up starting 1.5 s ahead, that first turn finished in 1.6 s.

Light turns also need a `low_local` group with a model in `groups`. Without it, every non-escalated
turn stays on the agent default.

`ceviz-pusula-snapshot` saves and restores named copies of this file
(`list`, `create <name>`, `restore <name>`, `mode [context_aware|single_turn|disabled]`).

## Privacy

Pusula runs inside the Ceviz helper process. It sees each command's text and the recent jobs'
texts and outcomes that Ceviz passes to routers. It calls only:

- `openclaw models list` on your machine
- the Ollama server at `decision_model_url` (on this machine by default)

## Development

```bash
python3 -m unittest discover -s tests -v
```

## Experiments

- [Escalation detector: regex vs. decision models](experiments/escalation-detector/README.md):
  a local comparison of the correction regex Pusula used until v0.1.0 with Kev 0.8B and 4B,
  Decider 2B, this-that 1.2, APUS-OpenJev-v1 4B and 9B, and qwen3:14b on 158 synthetic Turkish
  and English cases, plus first observations from production.

## License

MIT, see [LICENSE](LICENSE). The MIT license covers this code only. The Ceviz name, logo and icons
are not licensed by it; see Ceviz's
[trademark policy](https://github.com/MertBasar0/ceviz/blob/main/TRADEMARKS.md).
