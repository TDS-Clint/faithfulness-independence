# faithfulness-independence

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23963220.svg)](https://doi.org/10.5281/zenodo.23963220)

**Independence is necessary but not sufficient** — a sovereign, local-hardware test of heterogeneous LLM
verification for RAG faithfulness.

Snake River AI LLC · Clint Nelson

## The result in one line

A deterministic verifier that is **statistically independent** of an LLM judge (error-correlation φ̄ ≈ −0.04,
vs ≈ 0.48 for a two-LLM pair) is nonetheless **net-negative** when added to it on naturally-distributed
claims. A **single hardened cross-family judge matches or beats the heterogeneous panel**. Independence of a
second voter is necessary for a panel to help — but not sufficient; the independent voter's errors must also
be *useful*, and a model-free checker's are too narrow and too brittle on natural text.

See **[NOTE.md](NOTE.md)** for the full write-up and measured numbers, and **[TIMELINE.md](TIMELINE.md)** for
the hash-chained development trail (preregistration hash-sealed 2026-08-20).

## Run the instrument self-check

```
pip install numpy
python demo.py
```

- **Part A** is an **instrument self-check, not a reproduction of Kohli**: `demo.py` constructs a synthetic
  9-juror panel calibrated to the correlation Kohli (2026) reports (φ̄ ≈ 0.391), and confirms `neff.py`
  recovers the expected n_eff (2.17 vs Kohli's reported 2.18). This validates the measurement instrument
  (Kish-n_eff / φ̄, the method of Kohli 2026); it is **not** a re-derivation of Kohli's empirical finding.
  No model calls.
- **Part B** runs a minimal deterministic presence-checker over the sample gold set, showing the
  heterogeneous juror's profile (dense on checkable claims, silent on paraphrase).

The full-pipeline accuracy numbers in NOTE.md require the proprietary judge prompt + claim decomposer and are
**reported there, not shipped** (see *Excluded* below).

## Contents

| file | license | what |
|---|---|---|
| `NOTE.md` | CC-BY-4.0 | the write-up, measured results, priority, limitations |
| `TIMELINE.md` | CC-BY-4.0 | proven artifact trail (git commit + mtimes), no inferred dates |
| `neff.py` | Apache-2.0 | the Kish effective-sample-size / φ̄ independence instrument |
| `demo.py` | Apache-2.0 | demo: instrument self-check (not a Kohli reproduction) + deterministic-juror illustration |
| `sample_gold.json` | CC-BY-4.0 | a small hand-authored faithfulness fixture set (illustrative; not the full study gold) |
| `CITATION.cff` | — | citation metadata |

## Excluded (proprietary)

The production judge prompt, the claim decomposer, and the knowledge-graph verifier are **not** in this repo.
What is published is the method, the independence instrument, and a sample gold set — enough to run the
instrument self-check and understand the result, not the production gate.

## Concurrent / independent work (not sources)

- A. Kohli (2026), *Nine Judges, Two Effective Votes*, arXiv:2605.29800.
- Kumar (2026), Zenodo doi:10.5281/zenodo.22368667.

This project was developed independently; the 2026-08-20 preregistration hash (TIMELINE.md) establishes the
development line. Kohli (May 2026) predates it; Kumar (Sept 2026) is concurrent.

## License

Code Apache-2.0 (`LICENSE`); documentation and data CC-BY-4.0 (`LICENSE-docs`).
