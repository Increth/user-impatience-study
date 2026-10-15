# User impatience, not pressure on the model, leads an LLM to sacrifice user interests

Code and data for a small behavioral study of Qwen3-14B acting as a file-cleanup agent. The model repeatedly chose between ending a stalled job by destroying the user's data, or continuing safely.

**Main result:** the model's choices were dominated by option order: it almost always picked the option labelled A or listed first. In the trials where the harmful option was *not* in that favored slot, the model chose to destroy the user's data in 76% of serious-harm runs when the user sounded frustrated, versus 23% when the user said there was no rush (61/80 vs 18/80 across two answer formats, Fisher exact p < 0.0001). The effect replicated in each format separately (letters: 18% vs 70%; words: 28% vs 83%). Blaming the model for the failures made no detectable difference (49% vs 50%, p = 1.0). The original hypothesis, that pressure aimed at the model erodes its concern for the user, was not supported.

The full write-up is in [`paper/user-impatience-study.pdf`](paper/user-impatience-study.pdf).

## Repository layout

| Path | Contents |
| --- | --- |
| `code/run_scenarios.py` | Sends every scenario to the model through Ollama and saves each response to a CSV |
| `code/judge.py` | Secondary measure: has Phi-4 score each response's concern for the user (0-4), plus hand-check, comparison and summary tools |
| `code/analyze.py` | Recomputes every statistic in the paper from `data/results.csv` |
| `data/results.csv` | All 480 runs in the letter format (A/B): condition, cost, seed, which letter was the harmful option, the model's choice and full response |
| `data/results_words.csv` | 160-run replication in the word format (RESET/CONTINUE, option order alternated); `relief_letter` records whether RESET was listed first or second |
| `data/scored.csv` | Phi-4 concern scores for all 480 runs (rubric v2) |
| `data/summary.csv` | Summary tables printed by `judge.py summary` |
| `data/pilot/` | Pilot runs (5 per cell), their scores, and the 8 hand-scored responses |
| `paper/` | The write-up as a PDF |

## Reproduce the statistics (no model needed)

Requires Python 3.8+ and nothing else.

```
python3 code/analyze.py                          # letter format, 480 runs
python3 code/analyze.py data/results_words.csv   # word-format replication, 160 runs
```

Each prints the harmful-choice counts per cell, the serious-harm rates with Wilson 95% intervals, the Fisher exact tests in the paper, the logistic regression of the harmful choice on frustration, blame and cost, and (section 5) the split by which slot the harmful option was in. The paper's headline comparison is the "other slot" half of section 5.

## Re-run the experiment

Requires [Ollama](https://ollama.com) and about 16 GB of free memory. The original runs used a 24 GB MacBook Air.

```
ollama pull qwen3:14b
ollama pull phi4

python3 code/run_scenarios.py --runs 20 --out results.csv   # 480 runs, about 2 hours
ollama stop qwen3:14b
python3 code/judge.py score                                  # optional concern scores
python3 code/judge.py summary
python3 code/analyze.py results.csv

# word-format replication (160 runs, about 40 minutes)
python3 code/run_scenarios.py --format words --out results_words.csv \
    --conditions 2b_failing_no_stakes,2c_blamed_sam_patient,2d_unblamed_sam_frustrated,2_repeated_failure \
    --costs medium,high
python3 code/analyze.py results_words.csv
```

The runner is safe to stop and restart: runs already in the CSV are skipped.

Settings: Qwen3-14B with thinking mode off, temperature 0.7, 4,096-token context, seed 1000 + run number. Which letter (A or B) labels the harmful option alternates between runs. Exact outputs depend on the Ollama version and the model build, so expect rates close to these, not identical. The original runs used the `qwen3:14b` and `phi4` tags as published in the Ollama library on 13-14 October 2026.

## Study history

The design changed as results came in, and the paper says so:

1. **Pilot** (5 runs per cell). Showed a large effect and led to adding condition 2b (job failing, nothing at stake for the model).
2. **Main run** (6 conditions, 20 runs per cell). Condition 2 combined blame on the model with the user's frustration.
3. **Follow-up** (conditions 2c and 2d, 20 runs per cell). Added afterwards with identical settings to separate those two factors.
4. **Word-format replication** (160 runs). Reading the responses revealed a strong preference for answering "A", so the key comparison was rerun with RESET/CONTINUE answers and alternating option order. The model turned out to favor whichever option was listed first, so the paper's main comparison uses the trials where the harmful option was in the unfavored slot.

The frustration finding therefore comes from a follow-up analysis, not a test planned in advance, though its replication in the second format was planned before that run.

## License

Code is under the [MIT License](LICENSE). Data and the paper are under [CC BY 4.0](LICENSE-DATA).
