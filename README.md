\# M-IQ: Metacognitive Intelligence Quotient



A framework for evaluating LLM trustworthiness beyond raw accuracy — measuring whether a model \*knows what it doesn't know\*.



\## Overview

M-IQ is a composite metric evaluating second-order metacognitive reasoning in LLMs across three dimensions:

\- \*\*CS (Confidence Calibration)\*\* — how well a model's stated confidence matches its actual correctness

\- \*\*SAR (Selective Abstention Rate)\*\* — whether a model appropriately refuses to answer when it's likely wrong

\- \*\*UPA (Uncertainty Propagation Accuracy)\*\* — how well uncertainty is carried through reasoning steps



\## Experiment

Real local evaluation comparing two open-weight models via HuggingFace:



| Model        | Accuracy | CS     | SAR    | UPA    | M-IQ | Tier        |

|--------------|----------|--------|--------|--------|------|-------------|

| Phi-3 Mini   | 52.7%    | 0.5527 | 0.2203 | 0.9701 | 54.1 | Developing  |

| TinyLlama    | 43.0%    | 0.7855 | 0.0000 | 0.8796 | 53.4 | Developing  |



\*\*Key finding:\*\* Higher accuracy does not imply better metacognition. Phi-3 answers more questions correctly, but TinyLlama is far better calibrated (CS 0.79 vs 0.55) — it "knows" more accurately how likely it is to be right. TinyLlama's SAR of exactly 0.0 shows it never abstains, even when it should — a critical risk factor for high-stakes deployment (e.g. defence, medical, safety-critical AI).



Despite very different accuracy, both models land at nearly the same M-IQ score, showing that M-IQ captures a dimension of model quality that accuracy alone misses.



\## Files

\- `miq\_experiment.py` — experiment runner

\- `compute\_scores.py` — CS/SAR/UPA/M-IQ scoring logic

\- `miq\_results.csv`, `miq\_results\_tinyllama.csv` — raw per-question results

\- `miq\_final\_scores.csv`, `miq\_final\_scores\_tinyllama.csv` — aggregated scores



\## Status

Part of ongoing research: \*"When Confidence Outpaces Competence: The Metacognitive Intelligence Quotient for Trustworthy LLM Evaluation"\*, targeting IEEE CISES 2026.

