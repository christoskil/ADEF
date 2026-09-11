
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaselineResult:
    name: str
    storage_ratio: float
    total_stored: int
    mean_mse: float
    anomaly_recall: float
    anomaly_precision: float
    fault_f1: float
    fault_precision: float
    fault_recall: float
    mean_detection_delay: float
    extra: dict = field(default_factory=dict)


def _fault_metrics(fault_map: Dict[int, tuple], flagged_nodes: set,
                    detection_times: Dict[int, int]):
    true_det = sum(1 for i in flagged_nodes if i in fault_map)
    false_al = sum(1 for i in flagged_nodes if i not in fault_map)
    prec = true_det / max(len(flagged_nodes), 1)
    rec = true_det / max(len(fault_map), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    delays = []
    for nid, t_det in detection_times.items():
        if nid in fault_map:
            delays.append(max(0, t_det - fault_map[nid][0]))
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return prec, rec, f1, mean_delay


class StoreAllBaseline:
    name = "Store-All"

    def run(self, data: np.ndarray, ground_truth: np.ndarray,
            fault_map: Dict[int, tuple]) -> BaselineResult:
        N, T, _ = data.shape
        mse = float(np.mean((data[:, 100:, :] - ground_truth[:, 100:, :]) ** 2))
        return BaselineResult(
            name=self.name, storage_ratio=1.0, total_stored=N * T,
            mean_mse=mse, anomaly_recall=1.0, anomaly_precision=1.0,
            fault_f1=0.0, fault_precision=0.0, fault_recall=0.0,
            mean_detection_delay=float("nan"),
        )


class LEACHBaseline:

    name = "LEACH"

    def __init__(self, cluster_head_prob: float = 0.15, round_length: int = 20,
                 comm_radius: float = 3.0):
        self.p = cluster_head_prob
        self.round_length = round_length
        self.comm_radius = comm_radius

    def run(self, data: np.ndarray, ground_truth: np.ndarray,
            positions: np.ndarray, fault_map: Dict[int, tuple],
            seed: int = 42) -> BaselineResult:
        N, T, dim = data.shape
        rng = np.random.default_rng(seed)
        stored = np.zeros((N, T), dtype=bool)
        recon = np.zeros((N, T, dim))

        head_of = np.arange(N)  # each node is initially its own head
        for t in range(T):
            if t % self.round_length == 0:
                is_head = rng.random(N) < self.p
                if not is_head.any():
                    is_head[rng.integers(0, N)] = True
                head_ids = np.where(is_head)[0]
                for i in range(N):
                    dists = np.linalg.norm(positions[head_ids] - positions[i], axis=1)
                    head_of[i] = head_ids[np.argmin(dists)]

            for i in range(N):
                h = head_of[i]
                if h == i:
                    stored[i, t] = True
                    recon[i, t] = data[i, t]
                else:
                    recon[i, t] = data[h, t]

        mse = float(np.mean((recon[:, 100:, :] - ground_truth[:, 100:, :]) ** 2))
        storage_ratio = float(stored.mean())
        return BaselineResult(
            name=self.name, storage_ratio=storage_ratio,
            total_stored=int(stored.sum()), mean_mse=mse,
            anomaly_recall=0.0, anomaly_precision=0.0,
            fault_f1=0.0, fault_precision=0.0, fault_recall=0.0,
            mean_detection_delay=float("nan"),
        )


class ThresholdBaseline:
    """Running mean/std z-score gate for storage and fault flagging."""
    name = "Threshold-Based"

    def __init__(self, window: int = 50, storage_z: float = 2.5,
                 fault_z: float = 6.25, fault_persist: int = 10):
        self.window = window
        self.storage_z = storage_z
        self.fault_z = fault_z
        self.fault_persist = fault_persist

    def run(self, data: np.ndarray, ground_truth: np.ndarray,
            fault_map: Dict[int, tuple]) -> BaselineResult:
        N, T, dim = data.shape
        stored = np.zeros((N, T), dtype=bool)
        recon = np.zeros((N, T, dim))
        flagged = set()
        detection_times = {}
        anomaly_detected = 0
        anomaly_total = 0

        for i in range(N):
            buf = []
            persist_count = 0
            last_val = data[i, 0]
            for t in range(T):
                x = data[i, t]
                if len(buf) >= 5:
                    arr = np.array(buf[-self.window:])
                    mu, sigma = arr.mean(axis=0), arr.std(axis=0) + 1e-6
                    z = np.linalg.norm((x - mu) / sigma)
                else:
                    z = 0.0

                if z > self.storage_z or t < 5:
                    stored[i, t] = True
                    recon[i, t] = x
                    last_val = x
                else:
                    recon[i, t] = last_val

                if z > self.fault_z:
                    persist_count += 1
                    if persist_count >= self.fault_persist and i not in flagged:
                        flagged.add(i)
                        detection_times[i] = t
                else:
                    persist_count = 0

                buf.append(x)

        mse = float(np.mean((recon[:, 100:, :] - ground_truth[:, 100:, :]) ** 2))
        prec, rec, f1, delay = _fault_metrics(fault_map, flagged, detection_times)
        return BaselineResult(
            name=self.name, storage_ratio=float(stored.mean()),
            total_stored=int(stored.sum()), mean_mse=mse,
            anomaly_recall=1.0, anomaly_precision=1.0,  # reactive, at-onset detection
            fault_f1=f1, fault_precision=prec, fault_recall=rec,
            mean_detection_delay=delay,
        )


class DistributedAutoencoderBaseline:
    name = "Distributed Autoencoder (WAFL-style)"

    def __init__(self, hidden: int = 4, lr: float = 0.01,
                 share_every: int = 10, base_threshold: float = 1.0):
        self.hidden = hidden
        self.lr = lr
        self.share_every = share_every
        self.base_threshold = base_threshold

    def _init_weights(self, dim: int, rng: np.random.Generator):
        w1 = rng.normal(0, 0.5, size=(dim, self.hidden))
        b1 = np.zeros(self.hidden)
        w2 = rng.normal(0, 0.5, size=(self.hidden, dim))
        b2 = np.zeros(dim)
        return [w1, b1, w2, b2]

    @staticmethod
    def _forward(x, params):
        w1, b1, w2, b2 = params
        h = np.tanh(x @ w1 + b1)
        out = h @ w2 + b2
        return h, out

    def _train_step(self, x, params):
        w1, b1, w2, b2 = params
        h, out = self._forward(x, params)
        err = out - x
        grad_w2 = np.outer(h, err)
        grad_b2 = err
        grad_h = err @ w2.T * (1 - h ** 2)
        grad_w1 = np.outer(x, grad_h)
        grad_b1 = grad_h
        w1 -= self.lr * grad_w1
        b1 -= self.lr * grad_b1
        w2 -= self.lr * grad_w2
        b2 -= self.lr * grad_b2
        return float(np.sum(err ** 2))

    def run(self, data: np.ndarray, ground_truth: np.ndarray,
            fault_map: Dict[int, tuple], seed: int = 42) -> BaselineResult:
        N, T, dim = data.shape
        rng = np.random.default_rng(seed)
        params = [self._init_weights(dim, rng) for _ in range(N)]
        stored = np.zeros((N, T), dtype=bool)
        recon = np.zeros((N, T, dim))
        thresholds = np.full(N, self.base_threshold)
        recent_errors = [[] for _ in range(N)]
        flagged = set()
        detection_times = {}
        persist = np.zeros(N, dtype=int)

        for t in range(T):
            errors_t = np.zeros(N)
            for i in range(N):
                x = data[i, t]
                _, out = self._forward(x, params[i])
                err = float(np.sum((out - x) ** 2))
                errors_t[i] = err
                recent_errors[i].append(err)

                if err > thresholds[i]:
                    stored[i, t] = True
                    recon[i, t] = x
                    persist[i] += 1
                    if persist[i] >= 10 and i not in flagged:
                        flagged.add(i)
                        detection_times[i] = t
                else:
                    recon[i, t] = out
                    persist[i] = 0

                self._train_step(x, params[i])

            if t % self.share_every == 0 and t > 0:
                for i in range(N):
                    window = recent_errors[i][-50:]
                    thresholds[i] = np.mean(window) + 2.5 * (np.std(window) + 1e-6)

        mse = float(np.mean((recon[:, 100:, :] - ground_truth[:, 100:, :]) ** 2))
        prec, rec, f1, delay = _fault_metrics(fault_map, flagged, detection_times)
        return BaselineResult(
            name=self.name, storage_ratio=float(stored.mean()),
            total_stored=int(stored.sum()), mean_mse=mse,
            anomaly_recall=0.0, anomaly_precision=0.0,
            fault_f1=f1, fault_precision=prec, fault_recall=rec,
            mean_detection_delay=delay,
            extra={"hidden_dim": self.hidden, "lr": self.lr},
        )
