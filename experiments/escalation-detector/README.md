# Escalation detector: regex vs. decision models

Pusula moves a turn to a stronger model when the conversation shows that the previous attempt
missed. Today that signal comes from a fixed list of correction phrases plus a text-similarity
check for repeats. This experiment asks whether a decision model detects the signal better, and
how small local decision models compare with a general 14B model on the same task.

Everything here ran locally on one machine. The dataset is synthetic; no real conversation
content is included.

## Summary

- **The regex is the weakest option.** It misses every correction phrased in the user's own
  words (0/30) and escalates on new requests that merely contain a trigger word (22 of 25).
  Precision 0.53, recall 0.37.
- **qwen3:14b is the most accurate:** precision 0.97, recall 0.95, AUC 0.99 with the last three
  requests as context. It needs 10 GB of VRAM and about 0.24 s per turn for two questions.
- **this-that 1.2 separates the cases best among the small decision models** (AUC 0.96, ~4 GB,
  ~38 ms). Its probabilities are very small and shrink as the context grows, so its threshold has
  to be set low (about 0.03) and tuned per context length.
- **Decider 2B** has the most stable calibration across context lengths but separates the cases
  less well. **Kev 0.8B** is the weakest decision model, especially on speech-recognition errors.
- **More context helps qwen3:** three earlier requests were better than one or five.
- **Prompt order matters a lot:** putting the question before the state raised qwen3's false
  escalations from 2 to 29 and flipped 33 of 158 decisions.

## The task

The decision model replaces only the detection of the signal. The escalation level and the
model ladder stay in code, as they are in Pusula today.

**Evidence.** It is what Pusula already receives from the Ceviz helper (`recent_jobs`): the
last *N* finished requests with their outcome, status and age, plus the new message.

```json
{
  "recent_requests": [
    {"request": "Deploy durumunu kontrol et", "outcome": "blocked", "status": "completed", "minutes_ago": 11},
    {"request": "Hava nasıl?", "outcome": "done", "status": "completed", "minutes_ago": 8},
    {"request": "Saat kaç?", "outcome": "done", "status": "completed", "minutes_ago": 5}
  ],
  "new_message": "Deploy ne durumda, bir bakar mısın?"
}
```

**Questions.** Two Boolean questions are asked:

- **correction:** Does the new message tell the assistant that its previous attempt was wrong,
  incomplete, misunderstood the request, or did not work?
- **repeat:** Does the new message ask again for something a recent request already asked for,
  where that recent request did not end with outcome `done`?

**Decision.** A turn escalates when either probability reaches the threshold. The regex baseline
is the `escalation_level` Pusula used until v0.1.0, frozen in `regex_baseline.py`. Pusula now
asks APUS-OpenJev-v1 4B instead.

## Dataset

There are 158 synthetic cases in Turkish and English, built by [`dataset.py`](dataset.py) into
[`dataset.json`](dataset.json). Each case carries five earlier requests, all within Pusula's
15-minute window, and the runs truncate them to the last 1, 3 or 5. 86 cases should escalate and
72 should not.

| Category | Cases | Should escalate | Example |
|---|---|---|---|
| explicit_correction | 20 | yes | "Olmadı, iki test hâlâ kırık." / "That didn't work, two tests still fail." |
| paraphrased_correction | 30 | yes | "Hâlâ iki test kırık görünüyor." / "You changed the wrong file, the setting lives in config.yaml." |
| stt_correction | 12 | yes | "olmadi ya iki test hala kirik" |
| repeat_unsatisfied | 15 | yes | the same or a reworded request after a `blocked`/`unknown`/failed one |
| history_dependent | 13 | 9 yes, 4 no | the deciding request is three or four turns back |
| trigger_word_new_task | 25 | no | "README'deki yanlış linki bul ve değiştir." / "Add a try again button to the error screen." |
| non_correction_followup | 25 | no | "Teşekkürler, şimdi de lint'i çalıştır." |
| repeat_satisfied | 8 | no | the same request after one that ended `done` |
| negation_trap | 10 | no | "Yanlış anlaşılmasın, çok iyi olmuş; şimdi PR'ı aç." / "No need to try again, it's fine as is." |

## Models and conditions

| Model | Kind | Source | License | Weights | How it was queried |
|---|---|---|---|---|---|
| regex | Pusula until v0.1.0 | `regex_baseline.py` | MIT | – | `escalation_level` |
| Kev 0.8B | decision model | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | Apache-2.0 | LoRA on Qwen3.5-0.8B-Base | System One `noul`, `kev.serve` |
| Decider 2B | decision model | [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) | Apache-2.0 | 3.8 GB bf16 | System One `noul`, `decider/serve.py` |
| this-that 1.2 | decision model | [flock-io/this-that-model-1.2](https://huggingface.co/flock-io/this-that-model-1.2) | MIT | 3.8 GB bf16 (Decider 2B fine-tune) | System One `noul`, `thisthat.server` |
| qwen3:14b | general model | Ollama `qwen3:14b` | Apache-2.0 | 10.1 GB | A/B option letter, probability from first-token logprobs, thinking off |

**Hardware.** One RTX 5070 Ti (16 GB) under WSL2.

**GPU sharing.** The Ceviz helper's Whisper model stayed resident (about 2.5–4.3 GB). Each model
ran alone beside it; Kev's server was stopped while the others ran.

| Model | Approx. VRAM in use while serving | Latency p50 / p95, last 3 requests |
|---|---|---|
| Kev 0.8B | ~4 GB | 25 / 34 ms |
| Decider 2B | ~10 GB with its default CUDA-graph warm-up | 32 / 34 ms |
| this-that 1.2 | ~4 GB | 38 / 39 ms |
| qwen3:14b | 10.1 GB, fully on the GPU | 237 / 261 ms (two calls) |

**Software.** Ollama 0.20.6; torch 2.8 (CUDA 12.8); transformers 5.17.

**Excluded:**

- Hosted Jev: paid.
- Models that do not fit in 16 GB: OpenJev 27B, gemma3:27b, nemotron-3-nano:30b.
- gpt-oss:20b: reasoning model; its first token is not an answer.
- gemma4 e2b/e4b: under Ollama they always open a thinking channel first, so no first-token
  yes/no probability exists. An initial run that scored every case as "yes" was a measurement
  artifact and was discarded.

## Results

**Thresholds.** Each threshold was chosen once on the three-request runs and then applied to
every context length:

- regex: n/a
- Kev: 0.46
- Decider: 0.50
- this-that: 0.03
- qwen3: 0.50

### Last three requests as context

| Model | AUC | Threshold | Precision | Recall | False escalations | Missed escalations |
|---|---|---|---|---|---|---|
| regex | – | – | 0.53 | 0.37 | 28 | 54 |
| Kev 0.8B | 0.88 | 0.46 | 0.93 | 0.74 | 5 | 22 |
| Decider 2B | 0.90 | 0.50 | 0.84 | 0.76 | 12 | 21 |
| this-that 1.2 | 0.96 | 0.03 | 0.92 | 0.85 | 6 | 13 |
| **qwen3:14b** | **0.99** | 0.50 | **0.97** | **0.95** | **3** | **4** |

### By category, last three requests

Each cell counts correct decisions out of the cases in that category.

| Category | regex | Kev 0.8B | Decider 2B | this-that 1.2 | qwen3:14b |
|---|---|---|---|---|---|
| explicit_correction | 20/20 | 17/20 | 16/20 | 18/20 | 20/20 |
| paraphrased_correction | 0/30 | 22/30 | 19/30 | 25/30 | 28/30 |
| stt_correction | 5/12 | 1/12 | 6/12 | 7/12 | 10/12 |
| repeat_unsatisfied | 7/15 | 15/15 | 15/15 | 14/15 | 15/15 |
| history_dependent | 4/13 | 13/13 | 10/13 | 11/13 | 11/13 |
| trigger_word_new_task | 3/25 | 22/25 | 25/25 | 24/25 | 24/25 |
| non_correction_followup | 25/25 | 24/25 | 23/25 | 24/25 | 25/25 |
| repeat_satisfied | 8/8 | 7/8 | 2/8 | 8/8 | 8/8 |
| negation_trap | 4/10 | 10/10 | 9/10 | 8/10 | 10/10 |

### Context length (same thresholds)

| Model | Last 1 request: P / R | Last 3 requests: P / R | Last 5 requests: P / R |
|---|---|---|---|
| regex | 0.53 / 0.37 | 0.53 / 0.37 | 0.53 / 0.37 |
| Kev 0.8B | 0.78 / 0.78 | 0.93 / 0.74 | 0.92 / 0.70 |
| Decider 2B | 0.86 / 0.80 | 0.84 / 0.76 | 0.90 / 0.74 |
| this-that 1.2 | 0.90 / 0.91 | 0.92 / 0.85 | 0.92 / 0.65 |
| qwen3:14b | 0.93 / 0.88 | 0.97 / 0.95 | 0.98 / 0.90 |

The regex looks at every request in the 15-minute window regardless, so its numbers do not
change.

**Probability drift.** The decision models' probabilities fall as the state grows. The median
probability on cases that should escalate:

| Model | Last 1 | Last 3 | Last 5 |
|---|---|---|---|
| this-that 1.2 | 0.69 | 0.23 | 0.08 |
| Kev 0.8B | 0.52 | 0.50 | 0.49 |

That is why one fixed threshold does not hold across context lengths. Re-tuning per length
recovers most of it: for example, this-that at 0.01 on the last five requests gives precision
0.89 and recall 0.84.

### Prompt order (qwen3:14b, last five requests)

| Layout | Precision | Recall | False escalations | Missed escalations |
|---|---|---|---|---|
| State first, then question | 0.98 | 0.90 | 2 | 9 |
| Question first, then state | 0.74 | 0.97 | 29 | 3 |

33 of 158 decisions flipped between the two layouts, with a mean probability change of 0.21.
The decision-model servers apply their own trained layout, so this check applies to the general
model only.

### Earlier run, for reference

The first round used one earlier request and kept Kev resident. With that VRAM taken, qwen3:14b
ran partly on the CPU (4.9 of 10.2 GB on the GPU) and took 1.09 s per turn instead of 0.23 s. Its
accuracy was similar.

## What this means for Pusula

- A wrong escalation pins a stronger, paid model and switches off OpenClaw's fallback chain for
  that turn. A missed escalation usually comes back on the next turn as a repeat or a level-2
  miss. Precision therefore matters more than recall here.
- **If 10 GB can stay reserved:** qwen3:14b with the last three requests and the state first is
  the clear choice. Merging the two questions into one would halve the latency.
- **For a small, fast detector:** this-that 1.2 with the last one to three requests and a low
  threshold (about 0.03) is the best of the decision models tested. Its threshold should be tuned
  on real traffic first.
- **Before switching:** a shadow mode should run first. It logs the decision model's verdict next
  to the regex while Pusula still acts on the regex. The regex should stay as the fallback when
  the decision model times out.

## Limitations

- The dataset is synthetic. Its 158 cases were written and labelled by one person, and its
  category mix is deliberately harder than real traffic (it contains many regex traps).
- Thresholds were chosen on the same data they are reported on, so the tuned numbers are
  optimistic. AUC is threshold-free.
- There was one run per configuration. The model cards list this-that 1.2 as English-only, and
  results were not split by language.
- Latency was measured on one consumer GPU with Whisper resident. The numbers are not
  comparable with hosted Jev.

## Reproduce

Requires Python 3.11+ and this repository checked out.

```bash
cd experiments/escalation-detector
python3 dataset.py                      # writes dataset.json

# regex baseline (Pusula from this repository)
python3 escalation_eval.py regex --history 3

# any System One server (Kev, Decider, this-that); the model name is read from /v1/models
python3 escalation_eval.py systemone:http://127.0.0.1:8010 --history 3

# a general model through Ollama (state first; add :qfirst for question first)
python3 escalation_eval.py ollama:qwen3:14b --history 3

python3 summarize.py results/h3         # writes results/h3/summary.md
python3 compare_all.py                  # threshold sweep across models and lengths
```

Servers used:

```bash
python -m kev.serve --run jaredpalmer/kev-0.8b --host 127.0.0.1 --port 8009
DECIDER_MODEL=/path/to/decider-2b uvicorn decider.serve:app --host 127.0.0.1 --port 8010   # run from the model folder
python -m thisthat.server --model /path/to/this-that-model-1.2 --device cuda --host 127.0.0.1 --port 8011
```

The raw per-case results for every run are in [`results/`](results).
