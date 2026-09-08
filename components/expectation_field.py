"""
Component 1: Local Predictive Model (LPM)
==========================================
Implements the paper's Section 4.1 (Eqs. 1-6):
  - A Gaussian Mixture Model (GMM) as the distributional "Expectation
    Field", updated online via incremental EM (Eq. 1).
  - An Exponential Moving Average (EMA) point predictor for fast,
    step-ahead reconstruction (Eq. 2).
  - Mahalanobis-distance-based surprise scoring against the most
    probable GMM component (Eq. 3).
  - An adaptive threshold based on the coefficient of variation of
    recent deviations (Eq. 5), used for the binary storage decision
    (Eq. 4).

Default parameters match the tuned values used in the paper's
integrated pipeline (Section 5, Parameter Sensitivity discussion):
    n_components=3, alpha_gmm=0.05, eps_base=2.0, alpha_ema=0.30
"""

import numpy as np
from collections import deque
from typing import List, Tuple


class ExpectationField:
    """
    Per-node Local Predictive Model. One instance per sensor node.
    """

    def __init__(self, node_id: int, n_components: int = 3, dim: int = 2,
                 alpha: float = 0.05, eps_base: float = 2.0,
                 alpha_ema: float = 0.30, window: int = 50,
                 beta: float = 0.5, eta: float = 1e-3,
                 eps_adapt: bool = True, burn_in: int = 30):
        self.node_id = node_id
        self.K = n_components
        self.dim = dim
        self.alpha_gmm = alpha          # incremental EM learning rate
        self.eps_base = eps_base        # eps_0 in Eq. 5
        self._ema_alpha = alpha_ema     # alpha_EMA in Eq. 2
        self.W = window                 # window for threshold adaptation
        self.beta = beta                # amplitude of threshold adaptation
        self.eta = eta                  # regularization constant
        self.eps_adapt = eps_adapt
        self.burn_in = burn_in

        # GMM parameters: weights, means, covariances
        self.weights = np.ones(self.K) / self.K
        self.means = np.zeros((self.K, dim))
        self.covs = np.stack([np.eye(dim) * 4.0 for _ in range(self.K)])
        self._initialized = False

        # EMA point predictor
        self.ema = None

        # Bookkeeping
        self.t_seen = 0
        self.stored_steps: List[int] = []
        self.surprise_events: List[Tuple[int, float, np.ndarray]] = []
        self.predictions: List[np.ndarray] = []
        self.deviations: List[float] = []
        self.thresholds: List[float] = []
        self._recent_devs = deque(maxlen=self.W)

    # -- initialization -----------------------------------------------
    def _lazy_init(self, x: np.ndarray):
        """Seed the K means by small random perturbations of the first
        observation, so the mixture is not degenerate from step 0."""
        rng = np.random.default_rng(1000 + self.node_id)
        for k in range(self.K):
            self.means[k] = x + rng.normal(0, 0.5, size=self.dim)
        self._initialized = True

    # -- E-step: soft responsibilities ---------------------------------
    def _gauss_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        d = self.dim
        diff = x - mean
        cov_reg = cov + np.eye(d) * 1e-6
        inv = np.linalg.inv(cov_reg)
        det = max(np.linalg.det(cov_reg), 1e-12)
        exponent = -0.5 * diff @ inv @ diff
        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        return float(norm_const * np.exp(exponent))

    def _e_step(self, x: np.ndarray) -> np.ndarray:
        probs = np.array([self.weights[k] * self._gauss_pdf(x, self.means[k], self.covs[k])
                           for k in range(self.K)])
        total = probs.sum()
        if total <= 1e-12:
            return np.ones(self.K) / self.K
        return probs / total

    # -- incremental M-step (online EM, Eq. 1) --------------------------
    def _m_step_online(self, x: np.ndarray, r: np.ndarray):
        a = self.alpha_gmm
        for k in range(self.K):
            self.weights[k] = (1 - a) * self.weights[k] + a * r[k]
            diff = x - self.means[k]
            self.means[k] = self.means[k] + a * r[k] * diff
            outer = np.outer(diff, diff)
            self.covs[k] = (1 - a * r[k]) * self.covs[k] + a * r[k] * outer
        self.weights = self.weights / max(self.weights.sum(), 1e-12)

    # -- Mahalanobis deviation against best component (Eq. 3) -----------
    def _mahalanobis(self, x: np.ndarray, k: int) -> float:
        diff = x - self.means[k]
        cov_reg = self.covs[k] + np.eye(self.dim) * 1e-6
        inv = np.linalg.inv(cov_reg)
        val = diff @ inv @ diff
        return float(np.sqrt(max(val, 0.0)))

    # -- adaptive threshold (Eq. 5) --------------------------------------
    def _compute_threshold(self) -> float:
        if not self.eps_adapt or len(self._recent_devs) < max(5, self.W // 4):
            return self.eps_base
        arr = np.array(self._recent_devs)
        mean_d = arr.mean()
        std_d = arr.std()
        cv = std_d / (mean_d + self.eta)
        return self.eps_base * (1 + self.beta * np.tanh(cv - 1))

    # -- main per-step update --------------------------------------------
    def step(self, t: int, x: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Process one measurement. Returns (stored, deviation, ema_prediction).
        """
        if not self._initialized:
            self._lazy_init(x)
            self.ema = x.copy()

        # 1. EMA point prediction for *this* step (Eq. 2), based on
        #    the state carried over from t-1.
        pred = self.ema.copy()
        self.predictions.append(pred)

        # 2. E-step + Mahalanobis deviation against the best component
        r = self._e_step(x)
        best_k = int(np.argmax(r))
        deviation = self._mahalanobis(x, best_k)
        self._recent_devs.append(deviation)
        self.deviations.append(deviation)

        # 3. Adaptive threshold + storage decision (Eqs. 4-5)
        eps = self._compute_threshold()
        self.thresholds.append(eps)
        stored = (self.t_seen >= self.burn_in) and (deviation > eps)
        if stored:
            self.stored_steps.append(t)
            self.surprise_events.append((t, deviation, x.copy()))

        # 4. Update EMA predictor for the *next* step (Eq. 2)
        self.ema = self._ema_alpha * x + (1 - self._ema_alpha) * self.ema

        # 5. Online EM update (Eq. 1) -- the field always learns
        self._m_step_online(x, r)

        self.t_seen += 1
        return stored, deviation, pred

    # -- reconstruction & metrics ------------------------------------------
    def reconstruct(self, T: int, ground_truth: np.ndarray) -> np.ndarray:
        """
        Reconstructs the full series: stored steps use the true value,
        non-stored steps use the EMA prediction made at that step.
        """
        recon = np.array(self.predictions[:T])
        stored_set = set(self.stored_steps)
        for idx, t in enumerate(range(T)):
            if t in stored_set:
                recon[idx] = ground_truth[idx]
        return recon

    def storage_ratio(self, T: int) -> float:
        return len(self.stored_steps) / max(T, 1)

    def reconstruction_mse(self, ground_truth: np.ndarray, burn_in: int = 100) -> float:
        """MSE between reconstructed series and ground truth, skipping burn-in."""
        T = len(ground_truth)
        recon = self.reconstruct(T, ground_truth)
        return float(np.mean((recon[burn_in:T] - ground_truth[burn_in:T]) ** 2))
