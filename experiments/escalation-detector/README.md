# Escalation detector: regex vs. decision models

Pusula moves a turn to a stronger model when the conversation shows that the previous attempt
missed. Until v0.1.0 that signal came from a fixed list of correction phrases plus a
text-similarity check for repeats. This experiment asks whether a decision model detects the
signal better, and how small local decision models compare with general models on the same task.
Since v0.2.0, Pusula asks APUS-OpenJev-v1 4B, chosen from the third round below.

Everything here ran locally on one machine. The dataset is synthetic; no real conversation
content is included.

## Summary

- **The regex is the weakest option.** It misses every correction phrased in the user's own
  words (0/30) and escalates on new requests that merely contain a trigger word (22 of 25).
  Precision 0.53, recall 0.37.
- **APUS-OpenJev-v1 separates the cases best** (third round, last three requests). The 9B reaches
  AUC 1.00 (precision 0.94, recall 1.00 at 0.5; 11.3 GB, 0.40 s per turn for two questions). The
  4B reaches AUC 0.997 (precision 0.92, recall 0.99; 7 GB, 0.35 s). Their probabilities sit at
  either end, so 0.5 works without tuning. These are APUS AI Lab's OpenJev fine-tunes, not the
  OpenJev 27B model, which does not fit in 16 GB.
- **Kev 4B is the fastest strong option** (AUC 0.99, 93 ms, about 9.4 GB). At 0.5 it is precise
  but misses 11 of 86 escalations (precision 1.00, recall 0.87); at 0.31 it reaches precision
  0.94 and recall 0.98. The threshold is a call-site decision, not a property of the model.
- **qwen3:14b is the best general model:** precision 0.97, recall 0.95, AUC 0.99. It needs 10 GB
  and about 0.24 s per turn.
- **this-that 1.2 separates the cases best among the ~2B decision models** (AUC 0.96, ~4 GB,
  ~38 ms). Its probabilities are very small and shrink as the context grows, so its threshold has
  to be set low (about 0.03) and tuned per context length.
- **Decider 2B** has the most stable calibration across context lengths but separates the cases
  less well. **Kev 0.8B** is the weakest decision model, especially on speech-recognition errors.
- **Use at least three earlier requests.** With only one, every model misses most cases whose
  deciding request lies further back (4–7 of 13, against 10–13 of 13 with three).
- **Prompt order matters a lot:** putting the question before the state raised qwen3's false
  escalations from 2 to 29 and flipped 33 of 158 decisions.
- **In production** (five real turns so far), OpenJev 4B caught a direct pushback the regex had
  missed but also missed pushbacks that contradict a factual claim of the assistant, a kind of
  case this dataset does not contain. See [In production](#in-production-pusula-020).

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
| Kev 4B | decision model | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | Apache-2.0 | LoRA on Qwen3.5-4B-Base, bf16 | System One `noul`, `kev.serve` |
| OpenJev 4B (APUS) | decision model | [apus-ailab/APUS-OpenJev-v1-4B-GGUF](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF) | Apache-2.0 | Q8_0 GGUF, 4.5 GB | Ollama, same A/B prompt as qwen3; "yes"/"no" words counted for A/B |
| OpenJev 9B (APUS) | decision model | [apus-ailab/APUS-OpenJev-v1-9B-GGUF](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-9B-GGUF) | Apache-2.0 | Q8_0 GGUF, 9.5 GB | as the 4B |

The last three ran in a third round on 7 October.

**Hardware.** One RTX 5070 Ti (16 GB) under WSL2.

**GPU sharing.** The Ceviz helper's Whisper model stayed resident (about 2.5–4.3 GB). Each model
ran alone beside it; Kev's server was stopped while the others ran.

| Model | Approx. VRAM in use while serving | Latency p50 / p95, last 3 requests |
|---|---|---|
| Kev 0.8B | ~4 GB | 25 / 34 ms |
| Decider 2B | ~10 GB with its default CUDA-graph warm-up | 32 / 34 ms |
| this-that 1.2 | ~4 GB | 38 / 39 ms |
| qwen3:14b | 10.1 GB, fully on the GPU | 237 / 261 ms (two calls) |
| Kev 4B | ~9.4 GB | 93 / 115 ms |
| OpenJev 4B (APUS) | 7.0 GB at its default 9,216-token context (6.7 GB at 2,048) | 352 / 361 ms (two calls) |
| OpenJev 9B (APUS) | 11.3 GB | 400 / 410 ms (two calls) |

**Software.** Ollama 0.20.6; torch 2.8 (CUDA 12.8); transformers 5.17.

**Third-round scoring change.** APUS-OpenJev answers with the word "yes" or "no" about as often
as with the option letter, so the first-token probability now counts "yes" for A and "no" for B.
The qwen3:14b runs were scored before this change, on letters only.

**Excluded:**

- Hosted Jev: paid.
- Models that do not fit in 16 GB: OpenJev 27B (the original OpenJev model), gemma3:27b,
  nemotron-3-nano:30b.
- gpt-oss:20b: reasoning model; its first token is not an answer.
- gemma4 e2b/e4b: under Ollama they always open a thinking channel first, so no first-token
  yes/no probability exists. An initial run that scored every case as "yes" was a measurement
  artifact and was discarded.

## Results

**Thresholds.** Each threshold was chosen once on the three-request runs and then applied to
every context length:

- regex: n/a
- Kev 0.8B: 0.46
- Decider: 0.50
- this-that: 0.03
- qwen3: 0.50
- Kev 4B: 0.31 (at 0.50: precision 1.00, recall 0.87)
- OpenJev 4B and 9B: 0.50, untuned

### Last three requests as context

| Model | AUC | Threshold | Precision | Recall | False escalations | Missed escalations |
|---|---|---|---|---|---|---|
| regex | – | – | 0.53 | 0.37 | 28 | 54 |
| Kev 0.8B | 0.88 | 0.46 | 0.93 | 0.74 | 5 | 22 |
| Decider 2B | 0.90 | 0.50 | 0.84 | 0.76 | 12 | 21 |
| this-that 1.2 | 0.96 | 0.03 | 0.92 | 0.85 | 6 | 13 |
| qwen3:14b | 0.99 | 0.50 | 0.97 | 0.95 | 3 | 4 |
| Kev 4B | 0.99 | 0.31 | 0.94 | 0.98 | 5 | 2 |
| OpenJev 4B (APUS) | 0.997 | 0.50 | 0.92 | 0.99 | 7 | 1 |
| **OpenJev 9B (APUS)** | **1.00** | 0.50 | 0.94 | **1.00** | 6 | **0** |

### By category, last three requests

Each cell counts correct decisions out of the cases in that category.

| Category | regex | Kev 0.8B | Decider 2B | this-that 1.2 | qwen3:14b | Kev 4B | OpenJev 4B | OpenJev 9B |
|---|---|---|---|---|---|---|---|---|
| explicit_correction | 20/20 | 17/20 | 16/20 | 18/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| paraphrased_correction | 0/30 | 22/30 | 19/30 | 25/30 | 28/30 | 30/30 | 30/30 | 30/30 |
| stt_correction | 5/12 | 1/12 | 6/12 | 7/12 | 10/12 | 10/12 | 12/12 | 12/12 |
| repeat_unsatisfied | 7/15 | 15/15 | 15/15 | 14/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| history_dependent | 4/13 | 13/13 | 10/13 | 11/13 | 11/13 | 11/13 | 12/13 | 13/13 |
| trigger_word_new_task | 3/25 | 22/25 | 25/25 | 24/25 | 24/25 | 23/25 | 20/25 | 20/25 |
| non_correction_followup | 25/25 | 24/25 | 23/25 | 24/25 | 25/25 | 24/25 | 24/25 | 25/25 |
| repeat_satisfied | 8/8 | 7/8 | 2/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| negation_trap | 4/10 | 10/10 | 9/10 | 8/10 | 10/10 | 10/10 | 9/10 | 9/10 |

The OpenJev models' false escalations are mostly new requests that contain a trigger word
("find and fix the wrong link in the README").

### Context length (same thresholds)

| Model | Last 1 request: P / R | Last 3 requests: P / R | Last 5 requests: P / R |
|---|---|---|---|
| regex | 0.53 / 0.37 | 0.53 / 0.37 | 0.53 / 0.37 |
| Kev 0.8B | 0.78 / 0.78 | 0.93 / 0.74 | 0.92 / 0.70 |
| Decider 2B | 0.86 / 0.80 | 0.84 / 0.76 | 0.90 / 0.74 |
| this-that 1.2 | 0.90 / 0.91 | 0.92 / 0.85 | 0.92 / 0.65 |
| qwen3:14b | 0.93 / 0.88 | 0.97 / 0.95 | 0.98 / 0.90 |
| Kev 4B | 0.95 / 0.91 | 0.94 / 0.98 | 0.93 / 0.95 |
| OpenJev 4B (APUS) | 0.96 / 0.81 | 0.92 / 0.99 | 0.95 / 0.99 |
| OpenJev 9B (APUS) | 0.96 / 0.81 | 0.94 / 1.00 | 0.99 / 1.00 |

The regex looks at every request in the 15-minute window regardless, so its numbers do not
change. With only the last request, every model misses most cases whose deciding request lies
further back (4–7 of 13 history-dependent cases, against 10–13 of 13 with three). The OpenJev
models also missed 9 of the 15 repeats of an unfinished request with one request of context,
which every other model caught; with three they caught all 15.

**Probability drift.** The small decision models' probabilities fall as the state grows. The
median probability on cases that should escalate:

| Model | Last 1 | Last 3 | Last 5 |
|---|---|---|---|
| this-that 1.2 | 0.69 | 0.23 | 0.08 |
| Kev 0.8B | 0.52 | 0.50 | 0.49 |
| Kev 4B | 0.76 | 0.78 | 0.77 |
| OpenJev 4B and 9B | 0.999 | 0.999 | 0.999 |

That is why one fixed threshold does not hold across context lengths for the small models.
Re-tuning per length recovers most of it: for example, this-that at 0.01 on the last five
requests gives precision 0.89 and recall 0.84. Kev 4B and the OpenJev models stay stable.

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
- **Chosen for Pusula 0.2: OpenJev 4B (APUS)** with the last three requests, the state first and
  a fixed 0.5. It fits beside Whisper on a 16 GB card (6.7 GB with a 2,048-token context), needs
  no per-length tuning, and two questions take about 0.35 s. The 9B separates slightly better
  but needs 11.3 GB.
- **Kev 4B is the fast alternative** (93 ms) if its threshold is set for the call site; at the
  default 0.5 it misses one escalation in eight.
- **No regex fallback.** When the decision model does not answer within the turn's 1.2 s budget,
  the turn stays on the agent default. The owner chose this over a shadow-mode period.

## In production (Pusula 0.2.0)

A replay of five real turns from one evening, routed through the released code with the real
model, gave a first look outside the synthetic set:

- A direct pushback on an answer was caught (P = 0.99). The regex had missed it.
- Two pushbacks that contradict a factual claim of the assistant ("you can reach that machine",
  "we already set this up") were not caught (P = 0.14 and 0.13). The correction question asks
  whether the previous attempt was wrong or did not work, and the model did not read a disputed
  claim that way. The dataset has no such category yet.
- One turn that resumed an unfinished request was escalated (correction 0.95, repeat 0.52). That
  was not needed, but it was also not harmful.
- Latency was 0.38–0.62 s per turn, including the extra question asked only when a turn
  escalates and the previous one did not miss.

Two operational lessons:

- **Keep the model loaded.** A cold load took about 3.8 s, which does not fit the routing budget,
  so that turn fell back to the agent default. Pusula 0.2.1 keeps the model resident, and the
  Ceviz helper loads router plugins when its service starts.
- **The examples above are paraphrased.** Real conversation content is not published.

## Limitations

- The dataset is synthetic. Its 158 cases were written and labelled by one person, and its
  category mix is deliberately harder than real traffic (it contains many regex traps).
- Thresholds were chosen on the same data they are reported on, so the tuned numbers are
  optimistic. AUC is threshold-free.
- There was one run per configuration. The model cards list this-that 1.2 as English-only, and
  results were not split by language.
- The OpenJev models here are APUS AI Lab's fine-tunes, not the original OpenJev 27B. The third
  round counted "yes"/"no" words as answers; the qwen3:14b runs did not.
- The production observations cover five turns from one user.
- Latency was measured on one consumer GPU with Whisper resident. The numbers are not
  comparable with hosted Jev.

## Reproduce

Requires Python 3.11+ and this repository checked out.

```bash
cd experiments/escalation-detector
python3 dataset.py                      # writes dataset.json

# regex baseline (Pusula v0.1.0's regex, frozen in regex_baseline.py)
python3 escalation_eval.py regex --history 3

# any System One server (Kev, Decider, this-that); the model name is read from /v1/models
python3 escalation_eval.py systemone:http://127.0.0.1:8010 --history 3

# a model through Ollama (state first; add :qfirst for question first)
python3 escalation_eval.py ollama:qwen3:14b --history 3
python3 escalation_eval.py ollama:hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0 --history 3

python3 summarize.py results/h3         # writes results/h3/summary.md
python3 compare_all.py                  # threshold sweep across models and lengths
```

Servers used:

```bash
python -m kev.serve --run jaredpalmer/kev-0.8b --host 127.0.0.1 --port 8009
python -m kev.serve --run jaredpalmer/kev-4b --host 127.0.0.1 --port 8012
ollama pull hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0   # and the -9B-GGUF variant
DECIDER_MODEL=/path/to/decider-2b uvicorn decider.serve:app --host 127.0.0.1 --port 8010   # run from the model folder
python -m thisthat.server --model /path/to/this-that-model-1.2 --device cuda --host 127.0.0.1 --port 8011
```

The raw per-case results for every run are in [`results/`](results).
