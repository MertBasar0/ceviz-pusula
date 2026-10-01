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
- **It escalates after misses, deterministically.**
  - **Level 1:** the user corrects the answer ("anlamadın", "kastetmedim", "that's not what I
    meant", "didn't work"…), or repeats a request that did not finish. The turn runs on the best
    configured "balanced" model (for example the newest Sonnet) at `medium` thinking.
  - **Level 2:** a second miss within 15 minutes. The same model runs at `high` thinking.
  - A provider outage is not a miss, so it never escalates.
- **It only uses models you configured.** Pusula reads `openclaw models list --agent <agent> --json`
  and considers only models tagged as configured, default or a fallback, from Anthropic, OpenAI,
  Google and xAI. A model that is merely available may sit behind a paid API key you never chose
  for Ceviz.
- **Light tier (optional, off by default).** A local
  [System One](https://huggingface.co/spaces/multimodalart/jev-decision-index) decision server,
  such as Kev, can mark fresh small-talk turns for a lighter model.

## Install and enable

Pusula needs Python 3.11 or newer and a Ceviz helper that supports router contract v1. That
contract is not yet part of a published helper release; it is on Ceviz's
[`feat/router-plugin-contract`](https://github.com/MertBasar0/ceviz/tree/feat/router-plugin-contract)
branch.

```bash
# 1. Install into the helper's own environment
~/path/to/ceviz/.venv/bin/pip install git+https://github.com/MertBasar0/ceviz-pusula

# 2. Enable it for the helper service
systemctl --user edit watch-ceviz-backend
#   [Service]
#   Environment=WATCH_CEVIZ_ROUTER=pusula
systemctl --user restart watch-ceviz-backend

# 3. Check
~/path/to/ceviz/deploy/doctor.sh   # Router plugin "pusula" is installed and enabled
```

Ceviz's [router plugin guide](https://github.com/MertBasar0/ceviz/blob/feat/router-plugin-contract/docs/router-plugins.md)
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
| `escalation_window_seconds` | `900` | How long a miss counts toward the next level |
| `decision_endpoint` | unset | Local System One URL for the light tier, for example `http://127.0.0.1:8009` |
| `light_instructions`, `light_threshold` | unset, `0.8` | Question and probability threshold for light turns |

Light turns also need a `low_local` group with a model in `groups`. Without it, every non-escalated
turn stays on the agent default.

`ceviz-pusula-snapshot` saves and restores named copies of this file
(`list`, `create <name>`, `restore <name>`, `mode [context_aware|single_turn|disabled]`).

## Privacy

Pusula runs inside the Ceviz helper process. It sees each command's text and the recent jobs'
texts and outcomes that Ceviz passes to routers. It calls only:

- `openclaw models list` on your machine
- the `decision_endpoint` you configure, if any

## Development

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT, see [LICENSE](LICENSE). The MIT license covers this code only. The Ceviz name, logo and icons
are not licensed by it; see Ceviz's
[trademark policy](https://github.com/MertBasar0/ceviz/blob/main/TRADEMARKS.md).
