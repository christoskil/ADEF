

import numpy as np
from collections import deque
from typing import List, Dict


class CorrectnessTracker:

    def __init__(self, node_id: int, window: int = 80, lam: float = 0.03,
                 gamma: float = 2.0, rho_min: float = 0.3,
                 theta_rho: float = 1.5):
        self.node_id = node_id
        self.window = window
        self.lam = lam              # decay rate in w(tau, t) = exp(-lam*(t-tau))
        self.gamma = gamma          # relaxed correctness multiplier
        self.rho_min = rho_min      # floor for spatial agreement factor
        self.theta_rho = theta_rho  # consistency threshold for rho

        self._timestamps = deque(maxlen=window)
        self._correct_flags = deque(maxlen=window)
        self._agreement = deque(maxlen=window)
        self.confidence_history: List[float] = []

    def update(self, t: int, deviation: float, threshold: float,
               spatial_agreement: float = 1.0):

        correct = 1 if deviation <= (threshold * self.gamma) else 0
        self._correct_flags.append(correct)
        self._timestamps.append(t)
        self._agreement.append(np.clip(spatial_agreement, self.rho_min, 1.0))

        conf = self.confidence(t)
        self.confidence_history.append(conf)
        return conf

    def confidence(self, t: int) -> float:
        if not self._timestamps:
            return 0.5
        ts = np.array(self._timestamps)
        flags = np.array(self._correct_flags)
        agree = np.array(self._agreement)
        w = np.exp(-self.lam * (t - ts))
        num = np.sum(w * flags * agree)
        den = np.sum(w)
        return float(num / max(den, 1e-12))

    def compute_spatial_agreement(self, own_pred: np.ndarray,
                                   neighbor_readings: List[np.ndarray]) -> float:
        if not neighbor_readings:
            return self.rho_min
        agrees = [1 if np.linalg.norm(own_pred - xj) < self.theta_rho else 0
                  for xj in neighbor_readings]
        frac = sum(agrees) / len(agrees)
        return self.rho_min + (1 - self.rho_min) * frac


class ConfidencePropagator:


    def __init__(self, positions: np.ndarray, comm_radius: float = 2.0,
                 conflict_threshold: float = 1.5, blend_rate: float = 0.25):
        self.positions = positions
        self.N = positions.shape[0]
        self.comm_radius = comm_radius
        self.theta_div = conflict_threshold
        self.beta_blend = blend_rate
        self.neighbors = self._build_neighbor_graph()
        self.conflict_log: List[dict] = []

    def _build_neighbor_graph(self) -> Dict[int, List[int]]:
        neighbors = {i: [] for i in range(self.N)}
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.comm_radius:
                    neighbors[i].append(j)
        return neighbors

    def active_neighbors(self, i: int, quarantined: set) -> List[int]:
        return [j for j in self.neighbors[i] if j not in quarantined]

    def step(self, t: int, fields: list, trackers: List[CorrectnessTracker],
              quarantined: set = None):

        quarantined = quarantined or set()
        for i in range(self.N):
            if i in quarantined:
                continue
            for j in self.active_neighbors(i, quarantined):
                if j <= i:
                    continue  # each pair evaluated once
                pred_i, pred_j = fields[i].ema, fields[j].ema
                if np.linalg.norm(pred_i - pred_j) > self.theta_div:
                    self._resolve_conflict(t, i, j, fields, trackers)

    def _resolve_conflict(self, t: int, i: int, j: int, fields: list,
                           trackers: List[CorrectnessTracker]):
        c_i = trackers[i].confidence(t)
        c_j = trackers[j].confidence(t)
        if c_i == c_j:
            return
        winner, loser = (i, j) if c_i > c_j else (j, i)
        c_winner = max(c_i, c_j)
        c_loser = min(c_i, c_j)
        alpha_c = self.beta_blend * abs(c_winner - c_loser)

        f_winner, f_loser = fields[winner], fields[loser]
        f_loser.means = (1 - alpha_c) * f_loser.means + alpha_c * f_winner.means
        f_loser.covs = (1 - alpha_c) * f_loser.covs + alpha_c * f_winner.covs

        self.conflict_log.append({
            "t": t, "winner": winner, "loser": loser, "alpha_c": alpha_c
        })
