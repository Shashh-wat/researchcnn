"""Split conformal prediction wrapper (math.md §9). Post-hoc, no retraining required."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConformalCalibrator:
    alpha: float = 0.1  # target miscoverage epsilon (1-alpha = desired coverage)
    q_hat: float = None

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        scores: (n,) model's predicted P(y=1) for each calibration real image
        labels: (n,) ground-truth membership labels in {0,1}
        """
        n = len(scores)
        p_true_class = np.where(labels == 1, scores, 1 - scores)
        nonconformity = 1 - p_true_class
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(nonconformity, level, method="higher"))
        return self.q_hat

    def predict_set(self, score) -> list:
        """score: scalar or (n,) array, P(y=1). Returns prediction set(s) from {0,1}."""
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        score = np.atleast_1d(score)
        p0 = 1 - score
        p1 = score
        sets = []
        for a, b in zip(p0, p1):
            s = []
            if 1 - a <= self.q_hat:
                s.append(0)
            if 1 - b <= self.q_hat:
                s.append(1)
            sets.append(s)
        return sets if len(sets) > 1 else sets[0]
