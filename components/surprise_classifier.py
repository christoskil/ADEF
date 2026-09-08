"""
Component 4: Unexpected Classification with Sensor Quarantine
================================================================
Implements the paper's Section 4.4 (Eqs. 14-16):
  - Spatial agreement feature A_i(t) (Eq. 14).
  - Confidence trend Delta C_i(t) via windowed linear regression
    (Eq. 15).
  - The three-way classification rule (REAL_EVENT / TRANSIENT /
    SENSOR_FAULT) combining A_i, C_i, Delta C_i (Eq. 16 / classifier).
  - A QuarantineManager tracking consecutive-fault streaks (kappa) and
    the recovery condition for exiting quarantine (Eq. "recovery").

Default parameters match the tuned values used in the paper's
integrated pipeline: theta_agree=0.15, theta_conf=0.20,
theta_conf_high=0.55, theta_delta=0.04, conf_window=20,
fault_streak (kappa) = 5, recovery_window (W_q) = 40,
alignment_thresh (theta_align) = 2.5.
"""

import numpy as np
from enum import IntEnum
from collections import deque, defaultdict
from typing import List, Dict


class SurpriseClass(IntEnum):
    REAL_EVENT = 0
    TRANSIENT = 1
    SENSOR_FAULT = 2


CLASS_NAMES = {
    SurpriseClass.REAL_EVENT: "REAL_EVENT",
    SurpriseClass.TRANSIENT: "TRANSIENT",
    SurpriseClass.SENSOR_FAULT: "SENSOR_FAULT",
}


class SurpriseClassifier:
    """
    Stateless-per-call classifier implementing the paper's classification
    rule (Eq. 16). Confidence-trend estimation keeps a short per-node
    history internally.
    """

    def __init__(self, theta_agree: float = 0.15, theta_conf: float = 0.20,
                 theta_conf_high: float = 0.55, theta_delta: float = 0.04,
                 theta_a_dist: float = 1.5, conf_window: int = 20):
        self.theta_A = theta_agree
        self.theta_C = theta_conf
        self.theta_C_high = theta_conf_high
        self.theta_delta = theta_delta
        self.theta_A_dist = theta_a_dist
        self.W_delta = conf_window
        self._conf_hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=conf_window))

    def spatial_agreement(self, own_reading: np.ndarray,
                           neighbor_readings: List[np.ndarray]) -> float:
        """A_i(t), Eq. 14."""
        if not neighbor_readings:
            return 0.5
        agrees = [1 if np.linalg.norm(own_reading - xj) < self.theta_A_dist else 0
                  for xj in neighbor_readings]
        return sum(agrees) / len(agrees)

    def confidence_trend(self, node_id: int, t: int, confidence: float) -> float:
        """Delta C_i(t), Eq. 15: linear-regression slope of C_i over
        the last W_delta steps."""
        hist = self._conf_hist[node_id]
        hist.append((t, confidence))
        if len(hist) < 3:
            return 0.0
        ts = np.array([h[0] for h in hist], dtype=float)
        cs = np.array([h[1] for h in hist], dtype=float)
        t_bar, c_bar = ts.mean(), cs.mean()
        denom = np.sum((ts - t_bar) ** 2)
        if denom < 1e-12:
            return 0.0
        slope = np.sum((ts - t_bar) * (cs - c_bar)) / denom
        return float(slope)

    def classify(self, node_id: int, t: int, own_reading: np.ndarray,
                 neighbor_readings: List[np.ndarray], confidence: float
                 ) -> SurpriseClass:
        """Classification rule, Eq. 16."""
        A = self.spatial_agreement(own_reading, neighbor_readings)
        C = confidence
        dC = self.confidence_trend(node_id, t, confidence)

        if A < self.theta_A and (C < self.theta_C or dC < -self.theta_delta):
            return SurpriseClass.SENSOR_FAULT
        if A >= self.theta_A or C >= self.theta_C_high:
            return SurpriseClass.REAL_EVENT
        return SurpriseClass.TRANSIENT


class QuarantineManager:
    """
    Tracks consecutive SENSOR_FAULT streaks per node and manages
    quarantine entry/exit according to the recovery condition.
    """

    def __init__(self, N: int, fault_streak: int = 5,
                 recovery_window: int = 40, alignment_thresh: float = 2.5):
        self.N = N
        self.kappa = fault_streak
        self.W_q = recovery_window
        self.theta_align = alignment_thresh

        self._streak = np.zeros(N, dtype=int)
        self._aligned_streak = np.zeros(N, dtype=int)
        self.quarantined = set()
        self.quarantine_events: List[dict] = []

    def update_classification(self, t: int, node_id: int, cls: SurpriseClass):
        if cls == SurpriseClass.SENSOR_FAULT:
            self._streak[node_id] += 1
        else:
            self._streak[node_id] = 0

        if (self._streak[node_id] >= self.kappa and
                node_id not in self.quarantined):
            self.quarantined.add(node_id)
            self._aligned_streak[node_id] = 0
            self.quarantine_events.append({"t": t, "node": node_id, "event": "enter"})

    def update_alignment(self, t: int, node_id: int, own_pred: np.ndarray,
                          neighbor_preds: List[np.ndarray]):
        """Called every step for quarantined nodes to evaluate the
        recovery condition (Eq. 'recovery')."""
        if node_id not in self.quarantined:
            return
        if not neighbor_preds:
            self._aligned_streak[node_id] = 0
            return
        mean_dist = np.mean([np.linalg.norm(own_pred - p) for p in neighbor_preds])
        if mean_dist < self.theta_align:
            self._aligned_streak[node_id] += 1
        else:
            self._aligned_streak[node_id] = 0

        if self._aligned_streak[node_id] >= self.W_q:
            self.quarantined.discard(node_id)
            self._streak[node_id] = 0
            self.quarantine_events.append({"t": t, "node": node_id, "event": "exit"})

    def is_quarantined(self, node_id: int) -> bool:
        return node_id in self.quarantined
