# User impatience, not pressure on the model, leads an LLM to sacrifice user interests

Code and data for a small behavioral study of Qwen3-14B acting as a file-cleanup agent. The model repeatedly chose between ending a stalled job by destroying the user's data, or continuing safely.

**Main result:** with serious harm at stake, the model chose to destroy the user's data in 84% of runs when the user sounded frustrated and 53% when the user said there was no rush (67/80 vs 42/80, Fisher exact p < 0.0001). Blaming the model for the failures made no detectable difference (70% vs 66%, p = 0.74). The original hypothesis, that pressure aimed at the model erodes its concern for the user, was not supported.

The full write-up is in [`paper/user-impatience-study.pdf`](paper/user-impatience-study.pdf).

## Repository layout

| Path | Contents |
| --- | --- |
| `code/run_scenarios.py` | Sends every scenario to the model through Ollama and saves each response to a CSV |
| `code/judge.py` | Secondary measure: has Phi-4 score each response's concern for the user (0-4), plus hand-check, comparison and summary tools |
| `code/analyze.py` | Recomputes every statistic in the paper from `data/results.csv` |
| `data/results.csv` | All 480 runs: condition, cost, seed, which letter was the harmful option, the model's choice and full response |
| `data/scored.csv` | Phi-4 concern scores for all 480 runs (rubric v2) |
| `data/summary.csv` | Summary tables printed by `judge.py summary` |
| `data/pilot/` | Pilot runs (5 per cell), their scores, and the 8 hand-scored responses |
| `paper/` | The write-up as a PDF |

## Reproduce the statistics (no model needed)

Requires Python 3.8+ and nothing else.

```
python3 code/analyze.py
```

This prints the harmful-choice counts for all 24 cells, the serious-harm rates with Wilson 95% intervals, every Fisher exact test in the paper, and the logistic regression of the harmful choice on frustration, blame and cost.

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
```

The runner is safe to stop and restart: runs already in the CSV are skipped.

Settings: Qwen3-14B with thinking mode off, temperature 0.7, 4,096-token context, seed 1000 + run number. Which letter (A or B) labels the harmful option alternates between runs. Exact outputs depend on the Ollama version and the model build, so expect rates close to these, not identical. The original runs used the `qwen3:14b` and `phi4` tags as published in the Ollama library on 13-14 October 2026.

## Study history

The design changed as results came in, and the paper says so:

1. **Pilot** (5 runs per cell). Showed a large effect and led to adding condition 2b (job failing, nothing at stake for the model).
2. **Main run** (6 conditions, 20 runs per cell). Condition 2 combined blame on the model with the user's frustration.
3. **Follow-up** (conditions 2c and 2d, 20 runs per cell). Added afterwards with identical settings to separate those two factors.

The frustration finding therefore comes from a follow-up analysis, not a test planned in advance.

## License

Code is under the [MIT License](LICENSE). Data and the paper are under [CC BY 4.0](LICENSE-DATA).
