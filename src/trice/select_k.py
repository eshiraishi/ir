"""Choose the number of demonstrations k* from the training pool alone (no validation set).

Rule (pre-registered): micro-F1 of the leave-one-out curve over the k grid, smoothed by isotonic
regression (monotone, parameter-free), then the Kneedle knee (S=1, concave, increasing) on a
*linear* k axis, because prompt cost grows linearly in k. Without a knee, the smallest k reaching
the isotonic maximum. The 1-SE rule is reported for robustness only.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from kneed import KneeLocator
from sklearn.isotonic import IsotonicRegression

from trice.eval.metrics import f1_of_sums
from trice.eval.stats import bootstrap_se


@dataclass(frozen=True)
class KSelection:
    k_star: int
    rule: str  # "kneedle" or "isotonic-max"
    k_one_se: int
    ks: tuple[int, ...]
    f1: tuple[float, ...]
    f1_isotonic: tuple[float, ...]


def select_k(counts_by_k: Mapping[int, np.ndarray], seed: tuple = ()) -> KSelection:
    """``counts_by_k[k]`` holds the (n_queries, 3) LOO counts at k, aligned across k."""
    ks = sorted(counts_by_k)
    f1 = np.array([float(f1_of_sums(counts_by_k[k].sum(0))) for k in ks])
    iso = IsotonicRegression(increasing=True).fit_transform(np.array(ks, float), f1)

    knee = None
    if iso.max() > iso.min():
        knee = KneeLocator(ks, iso, S=1.0, curve="concave", direction="increasing").knee
    if knee is not None:
        k_star, rule = int(knee), "kneedle"
    else:
        k_star, rule = int(ks[int(np.flatnonzero(iso >= iso.max() - 1e-12)[0])]), "isotonic-max"

    best = int(np.argmax(f1))
    se = bootstrap_se(counts_by_k[ks[best]], seed=("one-se", *seed))
    k_one_se = int(ks[int(np.flatnonzero(f1 >= f1[best] - se)[0])])
    return KSelection(
        k_star, rule, k_one_se, tuple(ks), tuple(map(float, f1)), tuple(map(float, iso))
    )
