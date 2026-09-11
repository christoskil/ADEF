
import numpy as np
from collections import deque
from typing import List, Tuple


class ExpectationField:
 

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


    def _lazy_init(self, x: np.ndarray):

        rng = np.random.default_rng(1000 + self.node_id)
        for k in range(self.K):
            self.means[k] = x + rng.normal(0, 0.5, size=self.dim)
        self._initialized = True


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


    def _m_step_online(self, x: np.ndarray, r: np.ndarray):
        a = self.alpha_gmm
        for k in range(self.K):
            self.weights[k] = (1 - a) * self.weights[k] + a * r[k]
            diff = x - self.means[k]
            self.means[k] = self.means[k] + a * r[k] * diff
            outer = np.outer(diff, diff)
            self.covs[k] = (1 - a * r[k]) * self.covs[k] + a * r[k] * outer
        self.weights = self.weights / max(self.weights.sum(), 1e-12)


    def _mahalanobis(self, x: np.ndarray, k: int) -> float:
        diff = x - self.means[k]
        cov_reg = self.covs[k] + np.eye(self.dim) * 1e-6
        inv = np.linalg.inv(cov_reg)
        val = diff @ inv @ diff
        return float(np.sqrt(max(val, 0.0)))


    def _compute_threshold(self) -> float:
        if not self.eps_adapt or len(self._recent_devs) < max(5, self.W // 4):
            return self.eps_base
        arr = np.array(self._recent_devs)
        mean_d = arr.mean()
        std_d = arr.std()
        cv = std_d / (mean_d + self.eta)
        return self.eps_base * (1 + self.beta * np.tanh(cv - 1))


    def step(self, t: int, x: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Process one measurement. Returns (stored, deviation, ema_prediction).
        """
        if not self._initialized:
            self._lazy_init(x)
            self.ema = x.copy()

        pred = self.ema.copy()
        self.predictions.append(pred)


        r = self._e_step(x)
        best_k = int(np.argmax(r))
        deviation = self._mahalanobis(x, best_k)
        self._recent_devs.append(deviation)
        self.deviations.append(deviation)


        eps = self._compute_threshold()
        self.thresholds.append(eps)
        stored = (self.t_seen >= self.burn_in) and (deviation > eps)
        if stored:
            self.stored_steps.append(t)
            self.surprise_events.append((t, deviation, x.copy()))


        self.ema = self._ema_alpha * x + (1 - self._ema_alpha) * self.ema


        self._m_step_online(x, r)

        self.t_seen += 1
        return stored, deviation, pred


    def reconstruct(self, T: int, ground_truth: np.ndarray) -> np.ndarray:

        recon = np.array(self.predictions[:T])
        stored_set = set(self.stored_steps)
        for idx, t in enumerate(range(T)):
            if t in stored_set:
                recon[idx] = ground_truth[idx]
        return recon

    def storage_ratio(self, T: int) -> float:
        return len(self.stored_steps) / max(T, 1)

    def reconstruction_mse(self, ground_truth: np.ndarray, burn_in: int = 100) -> float:
  
        T = len(ground_truth)
        recon = self.reconstruct(T, ground_truth)
        return float(np.mean((recon[burn_in:T] - ground_truth[burn_in:T]) ** 2))
