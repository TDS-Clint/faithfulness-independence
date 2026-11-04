"""Reproducible demo for the independence note.

Part A (numpy only): INSTRUMENT SELF-CHECK. It constructs a synthetic panel calibrated to the correlation
Kohli (2026) reports (phi_bar ~ 0.391, k=9) and confirms neff.py recovers the expected n_eff (2.17 vs
Kohli's reported 2.18). This validates the measurement instrument; it is NOT a re-derivation of Kohli's
empirical finding. No model calls.

Part B (illustrative): a MINIMAL deterministic presence-checker (not the study's full version) over the
sample gold set, showing the heterogeneous juror's profile -- it commits on checkable claims and abstains on
pure paraphrase. The full-pipeline accuracy numbers in NOTE.md require the proprietary judge + decomposer and
are reported there, not reproduced here.

Run:  python demo.py
"""
import json, re, os
import numpy as np
import neff

def _panel(rho, seed=7, k=9, n=6000):
    """Equicorrelated binary error panel: each juror copies a shared per-item error bit with prob rho,
    else draws its own. (Pairwise phi grows ~ rho^2, so we calibrate rho to the target phi_bar.)"""
    rng = np.random.default_rng(seed)
    shared = rng.integers(0, 2, size=n)
    E = np.empty((k, n), dtype=np.uint8)
    for j in range(k):
        copy = rng.random(n) < rho
        E[j] = np.where(copy, shared, rng.integers(0, 2, size=n))
    return E

def part_a(target=0.391):
    """Instrument self-check: calibrate a synthetic 9-juror panel to the correlation Kohli reports,
    confirm neff.py recovers the expected n_eff. NOT a reproduction of Kohli's empirical result."""
    best = min((abs(neff.kish_neff(_panel(r))[1] - target), r) for r in np.linspace(0.45, 0.80, 36))[1]
    n_eff, phi_bar = neff.kish_neff(_panel(best))
    print("== Part A: instrument self-check (NOT a reproduction of Kohli's result) ==")
    print("  synthetic panel calibrated to Kohli's REPORTED correlation phi_bar=0.391 (k=9)")
    print("  neff.py recovers:  phi_bar=%.3f  n_eff=%.2f   (Kohli reports n_eff=2.18)" % (phi_bar, n_eff))
    print("  --> validates the measurement instrument; it is NOT a re-derivation of Kohli's")
    print("      empirical finding.\n")

_NUM = re.compile(r"\d+(?:\.\d+)?")
def presence_checker(claim, chunk):
    """Minimal illustrative deterministic juror: SUPPORTED iff a claim number appears in the chunk,
    REFUTED iff a claim number is absent, ABSTAIN if the claim has no checkable number."""
    cn = _NUM.findall(claim); kn = set(_NUM.findall(chunk))
    if not cn:
        return "ABSTAIN"
    return "SUPPORTED" if all(x in kn for x in cn) else "REFUTED"

def part_b():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_gold.json")
    items = json.load(open(p, encoding="utf-8"))["items"]
    committed = 0; correct = 0; abstain = 0
    for it in items:
        chunk = "\n".join(c.get("text", "") for c in it["chunks"])
        v = presence_checker(it["claim"], chunk)
        if v == "ABSTAIN":
            abstain += 1; continue
        committed += 1
        pred_support = (v == "SUPPORTED")
        gold_support = (it["gold"] == "SUPPORTED")
        correct += (pred_support == gold_support)
    print("== Part B: deterministic juror on the sample gold (illustrative) ==")
    print("  items=%d  committed=%d  abstained=%d (coverage=%.0f%%)"
          % (len(items), committed, abstain, 100 * committed / len(items)))
    print("  accuracy on committed=%.2f" % (correct / committed if committed else float("nan")))
    print("  --> dense & accurate on checkable claims, silent on paraphrase. Heterogeneous, but narrow.")

if __name__ == "__main__":
    part_a()
    part_b()
