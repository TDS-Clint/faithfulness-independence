# TIMELINE — hash-chained artifact trail

Proven artifacts only. No inferred dates. The anchor is a git commit; everything else is dated by git
history or filesystem mtime, and each entry says which.

## Anchor — hash-sealed preregistration (strongest evidence)

- **2026-08-20 11:38:42 −0600** — git commit **`870ca39e35c5354b6eab9ecbacc723d6fb28bbaa`**
  - message: `[PANEL] Preregistration sealed 2026-08-20T17:16:40.3591967Z`
  - body: *Panel-independence study: harness, bank v2 (150 items, 75/75), preregistration and hash record,
    sealed before any campaign ballot exists.*
  - sealed SHA-256 records committed in that commit:
    - `PREREGISTRATION.md` → `DB29BB259BC6854750B661EFC42FE873719CE8881BC9C56AF5DBCD19D2A0932F`
    - `ai_specs_v2.jsonl` → `BC1C3C545EFC5C81512D58F79F20E4EAE5F04BC2A1E7EED9F76C205956C3A2E9`
  - This is a **preregistration** — the independence study's protocol and item bank were hash-sealed *before*
    any ballot was run. It predates Kumar (Zenodo, Sept 2026).

## Forward chain — the faithfulness-gate work

- **2026-10-26 → 2026-10-31** — faithfulness-gate source files (segment/decompose/judge/recompose, gold
  harness) created, by filesystem mtime (`ku_v0/`). The faithfulness-gate thesis and gate are **October
  2026** work; they are **not** claimed as earlier.
- **2026-11-01 → 2026-11-04** — gate result runs, the natural-gold labeling, the measured-independence
  outcome (deterministic verifier net-negative; single cross-family judge wins), and the hash-chained
  failure-class ledger.

## Context (NOT claimed as evaluation work)

- **2026-05-11** — a local RAG server (`server.py`, logs) is present by filesystem mtime. This is
  **infrastructure context only** — a retrieval server standing up. It is **explicitly not** claimed as
  evaluation, independence, or faithfulness-gate work, and it carries no hash seal. It is listed only so the
  trail is complete and honest.

## What the trail proves, stated precisely

- A hash-sealed **judge-panel independence preregistration on 2026-08-20**, predating Kumar (Sept 2026).
- A **faithfulness-gate measurement line in October–November 2026**.
- No artifact supports a February or March 2026 origin, and none supports a "faithfulness-gate thesis by
  May 2026." Those dates were ideas, not artifacts, and are not claimed.

> Snake River AI's judge-panel independence preregistration is hash-sealed 2026-08-20, predating Kumar
> (Zenodo, Sept 2026). The measured-independence result — a deterministic verifier is statistically
> independent yet net-negative on natural text, and a single hardened cross-family judge matches or beats the
> panel — was developed independently on sovereign local hardware through Q3–Q4 2026. Kohli (arXiv, May 2026)
> predates this line of work; Kumar (Sept 2026) is concurrent. Development timeline substantiated by the
> hash-chained artifact trail in TIMELINE.md.
