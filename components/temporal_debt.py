"""
Component 3: Temporal Debt
============================
Implements the paper's Section 4.3 (Eqs. 11-13):
  - A three-state HMM (NORMAL, TRANSITION, ANOMALY) per node, updated
    online, decoded via the forward algorithm.
  - k-step-ahead ANOMALY probability forecasting (Eq. 12).
  - Debt issuance when the forecast exceeds theta_debt (TRANSITION
    state) or a reduced mu*theta_debt (early ANOMALY state), per the
    paper's Section 4.3 discussion of the dual trigger condition.
  - Debt acceptance by the neighbor with the lowest current debt load,
    subject to a per-node cooldown that bounds sensing overhead.

Default parameters match the tuned values used in the paper's
integrated pipeline: lookahead_k=20, debt_threshold=0.30,
max_debt_load=3.
"""

import numpy as np
from enum import IntEnum
from collections import deque, defaultdict
from typing import List, Dict, Tuple


class HMMState(IntEnum):
    NORMAL = 0
    TRANSITION = 1
    ANOMALY = 2


class DevLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class TemporalDebtManager:
    """
    Manages the per-node HMM state estimation and the network-wide
    issuance/acceptance/settlement of Temporal Debts.
    """

    def __init__(self, positions: np.ndarray, comm_radius: float = 2.0,
                 lookahead_k: int = 20, debt_threshold: float = 0.30,
                 mu: float = 0.5, max_debt_load: int = 3,
                 low_thresh: float = 1.0, high_thresh: float = 2.5,
                 forecast_horizon: int = 5):
        self.positions = positions
        self.N = positions.shape[0]
        self.comm_radius = comm_radius
        self.k = lookahead_k            # commitment horizon (t+k measurement)
        self.forecast_h = min(forecast_horizon, lookahead_k)  # probability forecast horizon
        self.theta_debt = debt_threshold
        self.mu = mu
        self.max_debt_load = max_debt_load
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

        self.neighbors = self._build_neighbor_graph()

        # HMM parameters per node (shared design, independently updated)
        self.A = [self._init_transition() for _ in range(self.N)]
        self.B = [self._init_emission() for _ in range(self.N)]
        self.pi = [np.array([0.9, 0.08, 0.02]) for _ in range(self.N)]
        self.alpha = [self.pi[i].copy() for i in range(self.N)]  # forward vector

        self.state_history: List[List[HMMState]] = [[] for _ in range(self.N)]
        self.debt_load = defaultdict(int)
        self.cooldown_until = defaultdict(int)
        self.debts_issued: List[dict] = []
        self.debts_confirmed = 0
        self.debts_total = 0

    def _build_neighbor_graph(self) -> Dict[int, List[int]]:
        neighbors = {i: [] for i in range(self.N)}
        for i in range(self.N):
            for j in range(self.N):
                if i != j and np.linalg.norm(
                        self.positions[i] - self.positions[j]) <= self.comm_radius:
                    neighbors[i].append(j)
        return neighbors

    @staticmethod
    def _init_transition() -> np.ndarray:
        # Designed prior: stay NORMAL most of the time, transitions are rare.
        return np.array([
            [0.95, 0.04, 0.01],
            [0.30, 0.50, 0.20],
            [0.10, 0.20, 0.70],
        ])

    @staticmethod
    def _init_emission() -> np.ndarray:
        # Rows = hidden state, cols = discretized deviation level (LOW/MED/HIGH)
        return np.array([
            [0.85, 0.13, 0.02],   # NORMAL -> mostly LOW deviation
            [0.30, 0.50, 0.20],   # TRANSITION -> mixed
            [0.05, 0.25, 0.70],   # ANOMALY -> mostly HIGH deviation
        ])

    def _discretize(self, deviation: float, threshold: float) -> DevLevel:
        ratio = deviation / max(threshold, 1e-6)
        if ratio < 0.75:
            return DevLevel.LOW
        elif ratio < 1.5:
            return DevLevel.MEDIUM
        return DevLevel.HIGH

    def _forward_update(self, i: int, obs: DevLevel):
        """One step of the forward algorithm (unnormalized -> normalized)."""
        A, B = self.A[i], self.B[i]
        pred = self.alpha[i] @ A
        likelihood = B[:, int(obs)]
        new_alpha = pred * likelihood
        total = new_alpha.sum()
        self.alpha[i] = new_alpha / total if total > 1e-12 else pred
        # light online adaptation of the emission row for the decoded state
        decoded = int(np.argmax(self.alpha[i]))
        B[decoded] = 0.97 * B[decoded]
        B[decoded, int(obs)] += 0.03
        B[decoded] /= B[decoded].sum()

    def predict_anomaly_prob(self, i: int, k: int = None) -> float:
        """P(ANOMALY_{t+k} | lambda_i), Eq. 12. Defaults to the shorter
        forecast_horizon (rather than the full commitment horizon) since
        forecasting many steps ahead washes out to the chain's stationary
        distribution and loses predictive power -- see Parameter
        Sensitivity discussion in the paper."""
        k = k or self.forecast_h
        Ak = np.linalg.matrix_power(self.A[i], k)
        probs = self.alpha[i] @ Ak
        return float(probs[HMMState.ANOMALY])

    def step(self, t: int, i: int, deviation: float, threshold: float
              ) -> Tuple[HMMState, float, bool]:
        """
        Process one measurement for node i. Returns (decoded_state,
        anomaly_probability, debt_issued).
        """
        obs = self._discretize(deviation, threshold)
        self._forward_update(i, obs)
        decoded = HMMState(int(np.argmax(self.alpha[i])))
        self.state_history[i].append(decoded)

        p_anomaly = self.predict_anomaly_prob(i)

        debt_issued = False
        if t >= self.cooldown_until[i]:
            trigger = False
            if decoded == HMMState.TRANSITION and p_anomaly >= self.theta_debt:
                trigger = True
            elif decoded == HMMState.ANOMALY and p_anomaly >= (self.mu * self.theta_debt):
                trigger = True

            if trigger:
                debt_issued = self._issue_debt(t, i)

        return decoded, p_anomaly, debt_issued

    def _issue_debt(self, t: int, issuer: int) -> bool:
        candidates = self.neighbors[issuer]
        if not candidates:
            return False
        # Acceptor = neighbor with the minimum current debt load.
        loads = [(self.debt_load[j], j) for j in candidates]
        loads.sort()
        acceptor = loads[0][1]
        if self.debt_load[acceptor] >= self.max_debt_load:
            return False

        self.debt_load[acceptor] += 1
        self.cooldown_until[issuer] = t + self.k
        self.debts_total += 1
        self.debts_issued.append({
            "t": t, "issuer": issuer, "acceptor": acceptor,
            "due": t + self.k, "confirmed": None,
        })
        return True

    def settle_due_debts(self, t: int, deviations_at_t: np.ndarray,
                          thresholds: np.ndarray):
        """Check debts due at time t; a debt is 'confirmed' if the
        acceptor's own deviation at t exceeds its threshold (i.e., the
        forecasted anomaly materialized), refuted otherwise."""
        for debt in self.debts_issued:
            if debt["due"] == t and debt["confirmed"] is None:
                acc = debt["acceptor"]
                confirmed = deviations_at_t[acc] > thresholds[acc]
                debt["confirmed"] = bool(confirmed)
                if confirmed:
                    self.debts_confirmed += 1
                self.debt_load[acc] = max(0, self.debt_load[acc] - 1)
