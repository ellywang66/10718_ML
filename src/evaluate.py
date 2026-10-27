"""Ranking metrics over the candidate set at one scoring date.

A ranking is an integer rank per candidate row (1 = inspect first). The top C rows are the C smallest ranks.
`inspected` and `serious` are boolean arrays over the same rows; `serious` is False wherever `inspected` is
False, because the outcome of an uninspected restaurant is unobserved. Every function takes C explicitly.
None of these metrics is designated primary here.
"""
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np


@dataclass
class RankingMetrics:
    """Metrics of one ranking's top C at one scoring date."""
    c: int
    n_top_inspected: int
    n_top_serious: int
    lower_bound_yield: float      # serious found in the top C / C, unlabeled rows count as misses
    conditional_yield: float      # serious / inspected, within the top C
    conditional_recall: float     # serious in the top C / all serious among inspected
    achd_yield: float             # all serious among inspected / C, the observed comparator
    unlabeled_share_top_c: float  # share of the top C with no label


def top_mask(rank: np.ndarray, c: int) -> np.ndarray:
    """Boolean mask of the C rows with the smallest rank (ties by position)."""
    top = np.zeros(len(rank), dtype=bool)
    top[np.argsort(rank, kind='stable')[:c]] = True
    return top


def metrics(rank: np.ndarray, inspected: np.ndarray, serious: np.ndarray, c: int) -> RankingMetrics:
    """All ranking metrics of the top C."""
    top = top_mask(rank, c)
    n_ins = int((inspected & top).sum())
    n_ser = int((serious & top).sum())
    all_ser = int(serious.sum())
    return RankingMetrics(
        c=c, n_top_inspected=n_ins, n_top_serious=n_ser,
        lower_bound_yield=n_ser / c,
        conditional_yield=n_ser / n_ins if n_ins else float('nan'),
        conditional_recall=n_ser / all_ser if all_ser else float('nan'),
        achd_yield=all_ser / c,
        unlabeled_share_top_c=1 - n_ins / c,
    )


def ipw_yield(rank: np.ndarray, inspected: np.ndarray, serious: np.ndarray, p: np.ndarray, c: int,
              floor: float) -> dict:
    """Inverse-propensity-weighted yield of the top C: sum of serious_i / p_i over inspected rows, divided by C.

    p is P(inspected | x) per row, clipped below at `floor`. Also returns the effective sample size of the
    weights over the inspected rows in the top C.
    """
    top = top_mask(rank, c)
    sel = top & inspected
    w = 1.0 / np.clip(p[sel], floor, None)
    ess = float(w.sum() ** 2 / (w ** 2).sum()) if sel.any() else 0.0
    return dict(ipw_yield=float((w * serious[sel]).sum() / c), n_weighted=int(sel.sum()), ess=ess,
                floor=floor, share_clipped=float((p[sel] < floor).mean()) if sel.any() else float('nan'))


def bootstrap(fn: Callable[[np.ndarray], float], n_rows: int, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap over candidate rows. fn maps an index array (rows with replacement) to a statistic."""
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n_rows, n_rows)) for _ in range(n)], dtype=float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return dict(se=float(np.nanstd(vals)), ci95=[float(lo), float(hi)])


def bootstrap_metric(name: str, rank: np.ndarray, inspected: np.ndarray, serious: np.ndarray, c: int,
                     n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap one field of RankingMetrics. Resampled rows keep their rank; the top C is recomputed."""
    def stat(idx: np.ndarray) -> float:
        return getattr(metrics(rank[idx], inspected[idx], serious[idx], c), name)
    return bootstrap(stat, len(rank), n, seed)


def to_dict(m: RankingMetrics) -> dict:
    """Plain dict for JSON output."""
    return asdict(m)
