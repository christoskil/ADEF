"""
Synthetic IoT Sensor Data Generator
====================================
Generates realistic 2D (temperature, humidity) sensor readings for a
network of N spatially-distributed nodes, following diurnal cycles with
spatially-correlated noise. Supports injection of real environmental
events (affecting all/nearby nodes) and sensor faults (affecting single
nodes), matching the experimental setup described in Section 5.1 of
the ADEF paper.

Default scenario (matches the paper):
  N = 20 nodes, 10x10 km area, sensing radius r = 2.0 km
  T = 2016 steps (7 days at 5-minute resolution)
  Events: fire (t=800-950, +8C, local), heat wave (t=1400-1600, +3C, global)
  Faults: spike (node, t=400-600), drift (node, t=1000-1200),
          dead sensor (node, t=1500-1700)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class NodeInfo:
    node_id: int
    x: float
    y: float
    fault_type: Optional[str] = None
    fault_start: Optional[int] = None
    fault_end: Optional[int] = None


@dataclass
class EventSpec:
    name: str
    kind: str          # "global" or "local"
    delta_temp: float
    start: int
    end: int
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None


class IoTDataGenerator:
    """
    Generates correlated (temperature, humidity) time series for a set
    of spatially deployed nodes, with diurnal cycles and injectable
    events / faults.
    """

    def __init__(self, nodes: List[NodeInfo], T: int,
                 base_temp: float = 22.0, temp_amplitude: float = 6.0,
                 base_humidity: float = 55.0, humidity_amplitude: float = 12.0,
                 steps_per_day: int = 288,  # 5-min resolution -> 288/day
                 spatial_noise_std: float = 0.4,
                 measurement_noise_std: float = 0.15,
                 seed: int = 42):
        self.nodes = nodes
        self.N = len(nodes)
        self.T = T
        self.base_temp = base_temp
        self.temp_amp = temp_amplitude
        self.base_hum = base_humidity
        self.hum_amp = humidity_amplitude
        self.steps_per_day = steps_per_day
        self.spatial_noise_std = spatial_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.rng = np.random.default_rng(seed)

    def _diurnal(self, t: int) -> Tuple[float, float]:
        phase = 2 * np.pi * (t % self.steps_per_day) / self.steps_per_day
        # Peak temperature mid-afternoon, peak humidity pre-dawn (inverse phase)
        temp = self.base_temp + self.temp_amp * np.sin(phase - np.pi / 2)
        hum = self.base_hum - self.hum_amp * np.sin(phase - np.pi / 2)
        return temp, hum

    def generate(self, events: List[EventSpec]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            data:         (N, T, 2) array of *measured* (possibly faulty)
                          readings, as would be received by each node.
            ground_truth: (N, T, 2) array of the true underlying signal
                          (no sensor faults injected -- used to score
                          reconstruction quality on healthy nodes).
        """
        N, T = self.N, self.T
        data = np.zeros((N, T, 2))
        ground_truth = np.zeros((N, T, 2))

        # One shared spatial random-walk offset per node (correlated drift
        # for nodes that are physically close), plus i.i.d. measurement noise.
        node_offset = self.rng.normal(0, 1.0, size=(N, 2))

        for t in range(T):
            base_temp, base_hum = self._diurnal(t)

            for i, node in enumerate(self.nodes):
                temp = base_temp + node_offset[i, 0] * 0.5
                hum = base_hum + node_offset[i, 1] * 0.5

                # Apply events (real environmental phenomena)
                for ev in events:
                    if ev.start <= t <= ev.end:
                        if ev.kind == "global":
                            temp += ev.delta_temp
                        elif ev.kind == "local" and ev.center is not None:
                            dist = np.hypot(node.x - ev.center[0],
                                             node.y - ev.center[1])
                            if dist <= (ev.radius or 0.0):
                                temp += ev.delta_temp

                # Spatially-correlated slow drift
                node_offset[i] += self.rng.normal(0, self.spatial_noise_std * 0.02, size=2)

                # Ground truth (no sensor fault applied)
                gt_reading = np.array([temp, hum]) + self.rng.normal(
                    0, self.measurement_noise_std, size=2)
                ground_truth[i, t, :] = gt_reading

                # Measured reading (fault applied if within this node's fault window)
                reading = gt_reading.copy()
                if (node.fault_type is not None and
                        node.fault_start is not None and
                        node.fault_start <= t <= (node.fault_end or T)):
                    reading = self._apply_fault(reading, node, t)

                data[i, t, :] = reading

        return data, ground_truth

    def _apply_fault(self, reading: np.ndarray, node: NodeInfo, t: int) -> np.ndarray:
        if node.fault_type == "spike":
            if self.rng.random() < 0.3:
                reading = reading + self.rng.normal(0, 8.0, size=2)
        elif node.fault_type == "drift":
            progress = (t - node.fault_start) / max(1, (node.fault_end - node.fault_start))
            reading = reading + np.array([10.0 * progress, 0.0])
        elif node.fault_type == "dead":
            reading = np.array([reading[0], reading[1]]) * 0.0 + np.array([0.0, 0.0])
        return reading


def build_default_scenario(N: int = 20, T: int = 2016, seed: int = 42,
                            area_km: float = 10.0, sensing_radius_km: float = 2.0
                            ) -> Tuple[IoTDataGenerator, List[EventSpec], List[NodeInfo]]:
    """
    Builds the default N=20, T=2016 scenario described in Section 5.1:
    a fire event, a heat wave, and three fault types (spike/drift/dead)
    injected into three distinct nodes; all other nodes remain healthy.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, area_km, size=(N, 2))

    nodes = [NodeInfo(node_id=i, x=positions[i, 0], y=positions[i, 1])
             for i in range(N)]

    # Inject three fault types into three distinct nodes (never node 0,
    # kept healthy as a reference node in plots)
    fault_node_ids = [1, 2, 3] if N > 3 else list(range(min(3, N)))
    fault_specs = [
        ("spike", 400, 600),
        ("drift", 1000, 1200),
        ("dead", 1500, 1700),
    ]
    for nid, (ftype, fstart, fend) in zip(fault_node_ids, fault_specs):
        nodes[nid].fault_type = ftype
        nodes[nid].fault_start = fstart
        nodes[nid].fault_end = min(fend, T - 1)

    center = (area_km / 2, area_km / 2)
    events = [
        EventSpec(name="fire", kind="local", delta_temp=8.0,
                  start=800, end=min(950, T - 1),
                  center=center, radius=2.0),
        EventSpec(name="heat_wave", kind="global", delta_temp=3.0,
                  start=1400, end=min(1600, T - 1)),
    ]

    gen = IoTDataGenerator(nodes=nodes, T=T, seed=seed)
    return gen, events, nodes
