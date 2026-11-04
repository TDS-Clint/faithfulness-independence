# Independence is necessary but not sufficient

**A sovereign, local-hardware test of heterogeneous LLM verification for RAG faithfulness**

Clint Nelson — Snake River AI LLC, Idaho Falls, Idaho
2026-11-04

---

## Abstract

Independence is necessary but not sufficient. We built the kind of LLM + deterministic *heterogeneous*
verifier that Kohli (2026) motivates and Kumar (2026) proposes, ran it on a retrieval-augmented-generation
(RAG) **faithfulness** task on local, sovereign hardware, and measured it against a labeled gold set. The
deterministic checker is **statistically independent** of the language-model judge (mean pairwise error
correlation φ̄ ≈ −0.04, versus φ̄ ≈ 0.48 for a two-LLM pair) — yet adding it to the judge is **net-negative**
on naturally-distributed claims. A **single hardened, cross-family judge matches or beats** the heterogeneous
panel. Independence bought the panel nothing it could spend: the deterministic voter's genuinely uncorrelated
errors do not translate into better decisions, because its coverage is narrow and its precision is brittle on
natural text. The contribution is the measurement and the negative result, reported in full.

## Priority

> Snake River AI's judge-panel independence preregistration is hash-sealed 2026-08-20, predating Kumar
> (Zenodo, Sept 2026). The measured-independence result — a deterministic verifier is statistically
> independent yet net-negative on natural text, and a single hardened cross-family judge matches or beats the
> panel — was developed independently on sovereign local hardware through Q3–Q4 2026. Kohli (arXiv, May 2026)
> predates this line of work; Kumar (Sept 2026) is concurrent. Development timeline substantiated by the
> hash-chained artifact trail in TIMELINE.md.

Kohli and Kumar are cited as **concurrent and independent** work, not as sources. This project did not derive
from either; the preregistration hash (TIMELINE.md) establishes the independent development line.

## Setup (method; production internals excluded)

- **Task:** faithfulness of a RAG answer to its cited passages — does a generated claim say only what the
  retrieved source asserts (extractive entailment), or does it add/derive/contradict?
- **Pipeline:** segment the answer into atomic claims → decompose compound claims → judge each unit against
  the retrieved passage union → recompose (strip-and-mark, with a whole-answer fail-closed fallback).
- **Judge (primary voter):** a ~27B-parameter local LLM from a **different model family** than the ~7B
  generator — cross-family by construction, to avoid shared-lineage error correlation. Temperature 0, local
  (no cloud).
- **Deterministic checker (heterogeneous second voter):** a model-free presence-checker — it extracts the
  claim's numbers, named entities and quoted/code spans and tests them against the cited passage, refuting
  only on a same-slot value mismatch. Zero model calls.
- **Second-LLM probe:** a third-family local LLM, evaluated as an alternative second voter.
- **Independence instrument:** Kish effective-sample-size `n_eff` and mean pairwise φ on committed ballots
  (`neff.py`, included), the method of Kohli (2026) so figures are directly comparable.
- **Gold:** a hand-authored fixture set (every known failure mode) plus a naturally-sourced set (claims drawn
  from the generator's actual answers over a query spread), labeled for faithfulness independently of the
  judge. Gate-vs-label disagreements are flagged for human review rather than scored silently.
- **Excluded from this release (proprietary):** the production judge prompt, the claim decomposer, and the
  knowledge-graph verifier. What is published is the method, the instrument, and a sample gold set.

## Results (measured on local hardware)

- **Hand-authored anchor (48 claims):** single hardened judge — false-accept 1, false-block 0,
  **accuracy 0.979**. The one false-accept is a cross-SDK contamination case handled by content-routing, not
  the judge.
- **Naturally-sourced set (144 scored; 12 faithful "punts" exempted):**
  - single hardened judge — FA 12, FB 8, **accuracy 0.861**;
  - heterogeneous panel (judge + deterministic checker) — FA 11, FB 17, **accuracy 0.806**.
  - The panel cuts 1 false-accept but adds 9 false-blocks: **net-negative**.
- **Independence:** deterministic-checker × judge φ̄ ≈ **−0.04** (statistically independent, even slightly
  anti-correlated) against φ̄ ≈ **0.48** for a two-LLM pair. Reference: Kohli reports **n_eff = 2.18** at
  **φ̄ = 0.391** for nine frontier judges (k=9). The deterministic voter is as independent as one could ask —
  and still does not help.
- **Second LLM does not rescue it:** a different-family second LLM caught only **4 of 17** of the residual
  errors the primary judge missed — it shares the primary judge's blind spot (consistent with Kohli's
  two-effective-votes finding).
- **Failure fingerprint** (judge-only, 63 withheld claims, hash-chained ledger):
  paraphrase 47 · invented-definition 5 · contamination 5 · value-swap 4 · punt 2.
- **Headline:** a single hardened cross-family judge matches or beats the heterogeneous panel. Statistical
  independence of a second voter is necessary for a panel to help, but not sufficient — the independent
  voter's errors must also be *useful* (dense coverage, high precision on the natural distribution), and the
  deterministic checker is not.

## Why the independent voter still loses

The deterministic checker's errors are genuinely uncorrelated with the judge's, but it is (a) silent on most
natural claims (pure paraphrase, where it abstains) and (b) brittle on natural formatting (markdown
enumerators, LaTeX, comma-adjacent numbers induce spurious refutals). Under a fail-closed consensus it can
only *add* withholds; on a clean fixture set those are rare, but on natural generated text they are frequent
and often wrong — so it trades a few caught fabrications for many withheld truths. Independence is real;
decision value is not.

## Limitations

- Single operator, single stack, **one corpus**; **not peer-reviewed**.
- Local hardware with specific open local models; results are for this configuration.
- Gold labels are author-produced (independent of the gate's judge, with disagreements flagged), not a
  multi-annotator consensus.
- The published harness is an **instrument self-check, not a reproduction of Kohli**: `demo.py` constructs a
  synthetic panel calibrated to the correlation Kohli reports, and confirms `neff.py` recovers the expected
  n_eff (2.17 vs Kohli's reported 2.18). This validates the measurement instrument; it is **not** a
  re-derivation of Kohli's empirical finding. The full-pipeline accuracy numbers above require the proprietary
  judge prompt + decomposer and are reported here, not shipped.

## References (concurrent / independent — not sources)

- A. Kohli (2026). *Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels.*
  arXiv:2605.29800.
- Kumar (2026). Zenodo, doi:10.5281/zenodo.22368667.

## License

Documentation (this file, README, TIMELINE): CC-BY-4.0. Code (`neff.py`, `demo.py`): Apache-2.0.
