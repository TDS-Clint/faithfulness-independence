"""
Panel independence metrics.

Measures how many genuinely independent votes a panel of judges actually
contains, using the Kish effective sample size framework.

Method follows Kohli (2026), "Nine Judges, Two Effective Votes: Correlated
Errors Undermine LLM Evaluation Panels" (arXiv:2605.29800), so that numbers
produced here are directly comparable to the published frontier-model results
(k=9, phi_bar=0.391, n_eff=2.18, Condorcet gap 22.0pp on ChaosNLI-MNLI).

Core quantities
---------------
error matrix E : (k, n) binary. E[j, i] = 1 iff judge j disagrees with gold on item i.
phi            : pairwise correlation of error vectors. For binary vectors the
                 phi coefficient equals the Pearson product-moment correlation,
                 which is what the Kish formula requires.
n_eff (Kish)   : k / (1 + (k-1) * phi_bar)
n_eff (eigen)  : k / lambda_max(phi_matrix)   -- drops the exchangeability assumption
Condorcet gap  : (accuracy if judges voted independently) - (actual panel accuracy)

Nothing here assumes the judges are LLMs. A juror is anything that emits a
label per item. That is deliberate: the whole point is to measure what happens
when one juror runs a structurally different procedure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

__all__ = [
    "error_matrix",
    "phi_matrix",
    "kish_neff",
    "eigen_neff",
    "bootstrap_neff",
    "permutation_test",
    "majority_vote",
    "condorcet_null",
    "leave_one_out",
    "swap_juror",
    "analyze_panel",
    "constant_jurors",
    "stratified_phi",
    "label_marginals",
    "phi_health",
    "PanelResult",
]


# --------------------------------------------------------------------------
# error vectors
# --------------------------------------------------------------------------

def error_matrix(votes: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Binary error matrix from a (k, n) vote matrix and length-n gold labels.

    Returns (k, n) uint8 where 1 == judge disagreed with gold.
    """
    votes = np.asarray(votes)
    gold = np.asarray(gold)
    if votes.ndim != 2:
        raise ValueError(f"votes must be 2-D (k judges, n items), got shape {votes.shape}")
    if votes.shape[1] != gold.shape[0]:
        raise ValueError(
            f"vote matrix has {votes.shape[1]} items but gold has {gold.shape[0]}"
        )
    return (votes != gold[None, :]).astype(np.uint8)


# --------------------------------------------------------------------------
# correlation and effective sample size
# --------------------------------------------------------------------------

def phi_matrix(E: np.ndarray) -> np.ndarray:
    """(k, k) matrix of pairwise phi coefficients between judge error vectors.

    A judge with zero error variance (always right or always wrong) has an
    undefined correlation with everyone; those cells come back NaN rather than
    being silently zero-filled, because zero would read as "independent" and
    that is exactly the wrong conclusion.
    """
    E = np.asarray(E, dtype=np.float64)
    k = E.shape[0]
    sd = E.std(axis=1)
    out = np.full((k, k), np.nan, dtype=np.float64)
    ok = sd > 0
    if ok.sum() >= 2:
        sub = np.corrcoef(E[ok])
        idx = np.where(ok)[0]
        out[np.ix_(idx, idx)] = sub
    np.fill_diagonal(out, 1.0)
    return out


def _mean_offdiag(M: np.ndarray) -> float:
    k = M.shape[0]
    iu = np.triu_indices(k, k=1)
    vals = M[iu]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def constant_jurors(E: np.ndarray, names: Sequence[str] | None = None) -> list:
    """Jurors with zero error variance -- always right, or always wrong.

    THIS IS A TRAP, and it is the most dangerous failure mode of the whole
    measurement. A juror that never varies has an UNDEFINED correlation with
    everyone else, not a zero one. Those NaN cells drop out of phi_bar while
    the juror is still counted in k, so n_eff goes UP and the panel looks more
    independent than it is.

    Concretely: a knowledge-graph checker asked about things outside its
    coverage abstains on every item. Add it to a panel and n_eff rises. That
    rise is an artifact of an abstaining juror, not evidence of procedural
    independence -- it would look exactly like the result we are hoping for
    while meaning nothing.

    Every entry point below calls this and refuses to return a number when it
    fires. A juror must actually disagree with gold sometimes, and agree with
    it sometimes, or it cannot be measured.
    """
    E = np.asarray(E)
    names = list(names) if names is not None else [f"juror_{i}" for i in range(E.shape[0])]
    out = []
    for j, nm in enumerate(names):
        rate = float(E[j].mean())
        if rate in (0.0, 1.0):
            out.append(
                {
                    "juror": nm,
                    "index": j,
                    "error_rate": rate,
                    "reason": "never wrong" if rate == 0.0 else "never right",
                }
            )
    return out


def _assert_measurable(E: np.ndarray, names: Sequence[str] | None = None) -> None:
    bad = constant_jurors(E, names)
    if bad:
        lines = "\n".join(
            f"    {b['juror']}: error_rate={b['error_rate']:.3f} ({b['reason']})" for b in bad
        )
        raise ValueError(
            "Cannot measure independence -- these jurors have zero error variance:\n"
            f"{lines}\n"
            "  Their correlation with the panel is UNDEFINED, not zero. Including\n"
            "  them would inflate n_eff and produce a false positive. Either drop\n"
            "  them from the panel or use a bank where they actually vary."
        )


def kish_neff(E: np.ndarray) -> tuple[float, float]:
    """Kish design-effect n_eff. Returns (n_eff, phi_bar).

    n_eff = k / (1 + (k-1) * phi_bar)

    Perfect independence (phi_bar = 0) gives n_eff = k.
    Perfect correlation (phi_bar = 1) gives n_eff = 1.
    The hard ceiling for any panel size is 1 / phi_bar.
    """
    E = np.asarray(E)
    k = E.shape[0]
    if k < 2:
        return float(k), float("nan")
    _assert_measurable(E)
    phi_bar = _mean_offdiag(phi_matrix(E))
    if np.isnan(phi_bar):
        return float("nan"), float("nan")
    denom = 1.0 + (k - 1) * phi_bar
    if denom <= 0:
        return float("nan"), phi_bar
    return k / denom, phi_bar


def eigen_neff(E: np.ndarray) -> tuple[float, float]:
    """Eigenvalue n_eff = k / lambda_max. Returns (n_eff, lambda_max).

    Robustness check on kish_neff: it makes no exchangeability assumption, so
    when the two agree closely the Kish number can be trusted.
    """
    E = np.asarray(E)
    k = E.shape[0]
    if k < 2:
        return float(k), 1.0
    M = phi_matrix(E)
    if np.isnan(M).any():
        M = np.nan_to_num(M, nan=0.0)
        np.fill_diagonal(M, 1.0)
    lam = float(np.linalg.eigvalsh(M).max())
    if lam <= 0:
        return float("nan"), lam
    return k / lam, lam


def bootstrap_neff(
    E: np.ndarray, n_boot: int = 10_000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for Kish n_eff, resampling items with replacement.

    This captures uncertainty over the item sample for *these specific judges*.
    It says nothing about how a different panel would score.
    """
    E = np.asarray(E)
    k, n = E.shape
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b], _ = kish_neff(E[:, idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------

def permutation_test(
    E: np.ndarray,
    strata: np.ndarray | None = None,
    n_perm: int = 10_000,
    seed: int = 42,
) -> dict:
    """Stratified permutation test on phi_bar.

    Within each difficulty stratum, each judge's error vector is shuffled
    independently. That preserves per-judge error rates and the item-difficulty
    structure while destroying inter-judge correlation. If the observed phi_bar
    still sits far above the permuted null, the correlation is real dependence
    and not just "these items were hard for everyone."
    """
    E = np.asarray(E)
    k, n = E.shape
    if strata is None:
        strata = np.zeros(n, dtype=int)
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    observed = _mean_offdiag(phi_matrix(E))
    rng = np.random.default_rng(seed)

    null = np.empty(n_perm, dtype=np.float64)
    Ep = E.copy()
    for p in range(n_perm):
        for g in groups:
            for j in range(k):
                Ep[j, g] = rng.permutation(E[j, g])
        null[p] = _mean_offdiag(phi_matrix(Ep))

    null = null[~np.isnan(null)]
    n_ge = int((null >= observed).sum())
    p_value = (n_ge + 1) / (null.size + 1)  # add-one, never reports p = 0
    sd = float(null.std())
    return {
        "observed_phi_bar": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": sd,
        "z": float((observed - null.mean()) / sd) if sd > 0 else float("nan"),
        "p_value": float(p_value),
        "n_perm": int(null.size),
    }


# --------------------------------------------------------------------------
# voting
# --------------------------------------------------------------------------

def _tiebreak(item_index: int, votes: Sequence, candidates: np.ndarray):
    """Deterministic SHA-256 tie-break, reproducible across runs and machines.

    Works for string labels as well as integer codes -- the real harness votes
    with strings like "SUPPORTED".
    """
    key = f"{item_index}:{','.join(str(v) for v in votes)}".encode()
    h = int(hashlib.sha256(key).hexdigest(), 16)
    return candidates[h % len(candidates)]


def majority_vote(votes: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
    """Majority vote across judges, deterministic SHA-256 tie-break."""
    votes = np.asarray(votes)
    k, n = votes.shape
    if classes is None:
        classes = np.unique(votes)
    classes = np.asarray(classes)
    out = np.empty(n, dtype=votes.dtype)
    for i in range(n):
        col = votes[:, i]
        counts = np.array([(col == c).sum() for c in classes])
        top = counts.max()
        winners = classes[counts == top]
        out[i] = winners[0] if winners.size == 1 else _tiebreak(i, col, winners)
    return out


# --------------------------------------------------------------------------
# Condorcet null model
# --------------------------------------------------------------------------

def condorcet_null(
    votes: np.ndarray,
    gold: np.ndarray,
    difficulty: np.ndarray | None = None,
    n_bins: int = 3,
    n_sim: int = 10_000,
    seed: int = 42,
) -> dict:
    """What would panel accuracy be if these judges voted *independently*?

    Each judge keeps its own per-difficulty-bin confusion matrix, so the
    simulation reproduces every judge's individual accuracy and class-specific
    error pattern exactly. The only thing removed is the correlation between
    them. The shortfall of actual accuracy below this prediction is the
    Condorcet gap: dependence beyond what shared item difficulty explains.
    """
    votes = np.asarray(votes)
    gold = np.asarray(gold)
    k, n = votes.shape
    classes = np.unique(np.concatenate([votes.ravel(), gold]))
    C = classes.size
    cls_index = {c: i for i, c in enumerate(classes)}

    if difficulty is None:
        bins = np.zeros(n, dtype=int)
    else:
        difficulty = np.asarray(difficulty, dtype=float)
        qs = np.quantile(difficulty, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(difficulty, qs)
    n_b = int(bins.max()) + 1

    gold_idx = np.array([cls_index[g] for g in gold])
    vote_idx = np.vectorize(cls_index.get)(votes)

    # per judge, per bin, per true class -> distribution over predicted class
    conf = np.zeros((k, n_b, C, C), dtype=np.float64)
    for j in range(k):
        for b in range(n_b):
            m = bins == b
            for c in range(C):
                sel = m & (gold_idx == c)
                if sel.sum() == 0:
                    conf[j, b, c] = np.full(C, 1.0 / C)  # unobserved cell -> uniform
                    continue
                counts = np.bincount(vote_idx[j][sel], minlength=C).astype(float)
                conf[j, b, c] = counts / counts.sum()

    # probability vector for each judge on each item
    P = np.empty((k, n, C), dtype=np.float64)
    for j in range(k):
        P[j] = conf[j, bins, gold_idx]
    cdf = np.cumsum(P, axis=2)

    rng = np.random.default_rng(seed)
    correct = 0
    chunk = max(1, min(n_sim, 2_000_000 // max(1, n * k)))
    done = 0
    while done < n_sim:
        s = min(chunk, n_sim - done)
        u = rng.random((k, n, s))
        sim = (u[..., None, :] > cdf[..., None]).sum(axis=2)  # (k, n, s)
        sim = np.clip(sim, 0, C - 1)
        counts = np.zeros((C, n, s), dtype=np.int16)
        for c in range(C):
            counts[c] = (sim == c).sum(axis=0)
        winner = counts.argmax(axis=0)  # ties -> lowest class index, unbiased in aggregate
        correct += int((winner == gold_idx[:, None]).sum())
        done += s

    predicted = correct / (n * n_sim)
    actual = float((majority_vote(votes, classes) == gold).mean())
    return {
        "predicted_accuracy": float(predicted),
        "actual_accuracy": actual,
        "condorcet_gap_pp": float((predicted - actual) * 100.0),
        "n_bins": int(n_b),
        "n_sim": int(n_sim),
    }


# --------------------------------------------------------------------------
# panel surgery
# --------------------------------------------------------------------------

def label_marginals(votes: np.ndarray, names: Sequence[str]) -> dict:
    """Each juror's label usage distribution.

    A first-class diagnostic, not a footnote. If one juror never emits a label
    the others use freely -- an LLM that never abstains sitting next to a
    knowledge-graph checker that abstains constantly -- then those jurors
    disagree systematically for reasons that have nothing to do with
    independent reasoning. Pooled phi cannot tell that apart from genuine
    error independence, so read this table before trusting any phi.
    """
    votes = np.asarray(votes)
    classes = np.unique(votes)
    out = {}
    for j, nm in enumerate(names):
        row = votes[j]
        out[nm] = {str(c): float((row == c).mean()) for c in classes}
    return out


def stratified_phi(
    votes: np.ndarray, gold: np.ndarray, names: Sequence[str]
) -> dict:
    """phi_bar and n_eff computed separately within each gold class.

    THIS IS THE PRIMARY NUMBER when jurors have different response biases.

    Pooled phi mixes two effects: whether jurors err on the same items, and
    whether they use the same parts of the label space. Within a single gold
    class every juror is attempting the same call, so marginal-usage
    differences largely drop out and what remains is genuine error overlap.

    A juror that decorrelates ONLY in the pooled number, and not within any
    class, has not demonstrated independence -- it has demonstrated a
    different default behaviour. Report both; lead with this one.

    Classes where a juror is constant are reported as unmeasurable rather than
    silently dropped.
    """
    votes = np.asarray(votes)
    gold = np.asarray(gold)
    names = list(names)
    E_all = error_matrix(votes, gold)
    out = {}
    for c in np.unique(gold):
        m = gold == c
        E = E_all[:, m]
        entry = {"n_items": int(m.sum())}
        bad = constant_jurors(E, names)
        entry["unmeasurable_jurors"] = bad
        entry["error_rates"] = {nm: float(E[j].mean()) for j, nm in enumerate(names)}
        if bad:
            entry["phi_bar"] = None
            entry["n_eff"] = None
            entry["note"] = (
                "not measurable within this class -- "
                + ", ".join(f"{b['juror']} {b['reason']}" for b in bad)
            )
        else:
            neff, phi = kish_neff(E)
            entry["phi_bar"] = float(phi)
            entry["n_eff"] = float(neff)
        out[str(c)] = entry
    return out


def phi_health(phi_bar: float, n_eff_kish: float, n_eff_eigen: float, k: int) -> list:
    """Warnings that must accompany any reported n_eff.

    Kish assumes non-negative intra-class correlation. Anti-correlated jurors
    return n_eff > k, which reads as 'more independent than independent' and is
    never a real result -- it is a signal that jurors are systematically
    disagreeing for structural reasons, almost always differing label-space
    usage.
    """
    w = []
    if phi_bar is not None and phi_bar < 0:
        w.append(
            f"phi_bar is NEGATIVE ({phi_bar:+.3f}). Kish assumes non-negative "
            f"correlation; n_eff is not interpretable. Jurors are systematically "
            f"ANTI-correlated. Check label_marginals() for response-bias asymmetry "
            f"before reading anything else. Do NOT report this as independence."
        )
    if n_eff_kish is not None and n_eff_kish > k + 1e-9:
        w.append(
            f"n_eff ({n_eff_kish:.2f}) EXCEEDS k ({k}). Impossible for genuinely "
            f"independent voting. Artifact of negative correlation."
        )
    if n_eff_kish and n_eff_eigen and max(n_eff_kish, n_eff_eigen) > 2 * min(
        n_eff_kish, n_eff_eigen
    ):
        w.append(
            f"Kish ({n_eff_kish:.2f}) and eigenvalue ({n_eff_eigen:.2f}) diverge "
            f"by more than 2x. Exchangeability is violated; pairwise correlations "
            f"are too heterogeneous for the Kish formula. Prefer the eigenvalue "
            f"estimate and report the full phi matrix."
        )
    return w


def leave_one_out(E: np.ndarray, names: Sequence[str]) -> list[dict]:
    """Effect on n_eff of dropping each juror.

    A positive delta means the panel gets *more* independent without that
    juror, i.e. its errors were the most redundant with everyone else's.
    """
    E = np.asarray(E)
    base, _ = kish_neff(E)
    rows = []
    for j, name in enumerate(names):
        keep = [i for i in range(E.shape[0]) if i != j]
        neff, phi = kish_neff(E[keep])
        rows.append(
            {
                "removed": name,
                "n_eff_without": float(neff),
                "delta_n_eff": float(neff - base),
                "phi_bar_without": float(phi),
            }
        )
    return sorted(rows, key=lambda r: -r["delta_n_eff"])


def swap_juror(
    E: np.ndarray, names: Sequence[str], out_name: str, in_name: str
) -> dict:
    """Compare the panel with `out_name` against the panel with `in_name`.

    This is the measurement the literature leaves open. Kohli shows that
    swapping model brands barely moves n_eff -- restricting to one judge per
    family actually *lowered* it to 1.93. The open question is whether a juror
    running a structurally different procedure (a grounded claim checker, a
    test executor, a retrieval-backed verifier) buys real independence where
    another language model does not.

    Positive delta_n_eff means the dissimilar juror bought independence that
    a language model could not.
    """
    names = list(names)
    for nm in (out_name, in_name):
        if nm not in names:
            raise ValueError(f"{nm!r} not in panel: {names}")
    oi, ii = names.index(out_name), names.index(in_name)

    with_out = [i for i in range(len(names)) if i != ii]
    with_in = [i for i in range(len(names)) if i != oi]

    neff_out, phi_out = kish_neff(E[with_out])
    neff_in, phi_in = kish_neff(E[with_in])

    return {
        "baseline_juror": out_name,
        "swapped_juror": in_name,
        "n_eff_baseline": float(neff_out),
        "n_eff_swapped": float(neff_in),
        "delta_n_eff": float(neff_in - neff_out),
        "phi_bar_baseline": float(phi_out),
        "phi_bar_swapped": float(phi_in),
        "delta_phi_bar": float(phi_in - phi_out),
    }


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

@dataclass
class PanelResult:
    names: list[str] = field(default_factory=list)
    k: int = 0
    n_items: int = 0
    error_rates: dict = field(default_factory=dict)
    phi_bar: float = float("nan")
    phi_min: float = float("nan")
    phi_max: float = float("nan")
    n_eff_kish: float = float("nan")
    n_eff_ci: tuple = (float("nan"), float("nan"))
    n_eff_eigen: float = float("nan")
    lambda_max: float = float("nan")
    independence_ratio: float = float("nan")
    ceiling: float = float("nan")
    panel_accuracy: float = float("nan")
    best_individual: str = ""
    best_individual_accuracy: float = float("nan")
    panel_lift_pp: float = float("nan")
    all_wrong_items: int = 0
    all_right_items: int = 0
    label_marginals: dict = field(default_factory=dict)
    stratified: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    condorcet: dict = field(default_factory=dict)
    permutation: dict = field(default_factory=dict)
    leave_one_out: list = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_panel(
    votes: np.ndarray,
    gold: np.ndarray,
    names: Sequence[str],
    difficulty: np.ndarray | None = None,
    n_boot: int = 2_000,
    n_perm: int = 2_000,
    n_sim: int = 2_000,
    seed: int = 42,
) -> PanelResult:
    """Full diagnostic on a panel.

    Defaults for n_boot / n_perm / n_sim are lighter than the paper's 10,000 so
    an interactive run finishes quickly. Turn them up for a result you intend
    to publish.
    """
    votes = np.asarray(votes)
    gold = np.asarray(gold)
    names = list(names)
    k, n = votes.shape
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} judges")

    E = error_matrix(votes, gold)
    _assert_measurable(E, names)
    neff, phi_bar = kish_neff(E)
    neff_e, lam = eigen_neff(E)
    lo, hi = bootstrap_neff(E, n_boot=n_boot, seed=seed)

    M = phi_matrix(E)
    iu = np.triu_indices(k, k=1)
    offdiag = M[iu]
    offdiag = offdiag[~np.isnan(offdiag)]

    acc = 1.0 - E.mean(axis=1)
    best = int(np.argmax(acc))
    panel_acc = float((majority_vote(votes) == gold).mean())

    per_item = E.sum(axis=0)

    strata = None
    if difficulty is not None:
        d = np.asarray(difficulty, dtype=float)
        qs = np.quantile(d, [1 / 3, 2 / 3])
        strata = np.digitize(d, qs)

    res = PanelResult(
        names=names,
        k=k,
        n_items=int(n),
        error_rates={nm: float(E[j].mean()) for j, nm in enumerate(names)},
        phi_bar=float(phi_bar),
        phi_min=float(offdiag.min()) if offdiag.size else float("nan"),
        phi_max=float(offdiag.max()) if offdiag.size else float("nan"),
        n_eff_kish=float(neff),
        n_eff_ci=(lo, hi),
        n_eff_eigen=float(neff_e),
        lambda_max=float(lam),
        independence_ratio=float(neff / k),
        ceiling=float(1.0 / phi_bar) if phi_bar and phi_bar > 0 else float("inf"),
        panel_accuracy=panel_acc,
        best_individual=names[best],
        best_individual_accuracy=float(acc[best]),
        panel_lift_pp=float((panel_acc - acc[best]) * 100.0),
        all_wrong_items=int((per_item == k).sum()),
        all_right_items=int((per_item == 0).sum()),
        label_marginals=label_marginals(votes, names),
        stratified=stratified_phi(votes, gold, names),
        warnings=phi_health(float(phi_bar), float(neff), float(neff_e), k),
        condorcet=condorcet_null(votes, gold, difficulty, n_sim=n_sim, seed=seed),
        permutation=permutation_test(E, strata=strata, n_perm=n_perm, seed=seed),
        leave_one_out=leave_one_out(E, names),
    )

    ratio = res.independence_ratio
    if ratio < 0.5:
        res.verdict = (
            f"FAILS the n_eff/k >= 0.5 diagnostic. {k} jurors carry "
            f"{neff:.2f} independent votes' worth of information. Treat "
            f"agreement from this panel with caution -- it is largely one "
            f"opinion counted {k} times."
        )
    elif ratio < 0.75:
        res.verdict = (
            f"Marginal. {k} jurors carry {neff:.2f} independent votes. There "
            f"is real but limited diversity here."
        )
    else:
        res.verdict = (
            f"Healthy. {k} jurors carry {neff:.2f} independent votes "
            f"({ratio:.0%} of nominal). Errors are close to independent."
        )
    return res
