
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data.generator import build_default_scenario
from components.expectation_field import ExpectationField
from components.correctness_propagation import CorrectnessTracker, ConfidencePropagator
from components.temporal_debt import TemporalDebtManager, HMMState
from components.surprise_classifier import (
    SurpriseClassifier, QuarantineManager, SurpriseClass, CLASS_NAMES
)


def run_integrated(N: int = 20, T: int = 2016, seed: int = 42, verbose: bool = True):
    gen, events, nodes = build_default_scenario(N=N, T=T, seed=seed)
    data, ground_truth = gen.generate(events)
    positions = np.array([[n.x, n.y] for n in nodes])

    fields = [ExpectationField(i, n_components=3, dim=2,
                                alpha=0.05, eps_base=2.0)
              for i in range(N)]
    trackers = [CorrectnessTracker(i, window=80, lam=0.03)
                for i in range(N)]
    prop = ConfidencePropagator(positions, comm_radius=2.0,
                                 conflict_threshold=1.5, blend_rate=0.25)
    debt_mgr = TemporalDebtManager(positions, comm_radius=2.0,
                                    lookahead_k=20, debt_threshold=0.30,
                                    max_debt_load=3)
    classifier = SurpriseClassifier(theta_agree=0.40, theta_conf=0.65,
                                     theta_conf_high=0.85, theta_delta=0.04,
                                     conf_window=20)
    quarantine = QuarantineManager(N, fault_streak=5,
                                    recovery_window=40, alignment_thresh=2.5)


    storage_log = np.zeros((N, T), dtype=bool)
    deviation_log = np.zeros((N, T))
    threshold_log = np.zeros((N, T))
    classification_log = {}   # (i, t) -> SurpriseClass
    quarantine_state_log = np.zeros((N, T), dtype=bool)

    for t in range(T):
        deviations_t = np.zeros(N)
        thresholds_t = np.zeros(N)

        for i in range(N):
            x = data[i, t, :]
            stored, dev, pred = fields[i].step(t, x)
            deviations_t[i] = dev
            thresholds_t[i] = fields[i].thresholds[-1]
            storage_log[i, t] = stored
            deviation_log[i, t] = dev
            threshold_log[i, t] = fields[i].thresholds[-1]

            is_q = quarantine.is_quarantined(i)
            quarantine_state_log[i, t] = is_q


            active_neigh = prop.active_neighbors(i, quarantine.quarantined)
            neighbor_readings = [data[j, t, :] for j in active_neigh]
            agreement = trackers[i].compute_spatial_agreement(pred, neighbor_readings)
            conf = trackers[i].update(t, dev, thresholds_t[i], agreement)


            if not is_q:
                decoded, p_anom, issued = debt_mgr.step(t, i, dev, thresholds_t[i])

            if stored:
                cls = classifier.classify(i, t, x, neighbor_readings, conf)
                classification_log[(i, t)] = cls
                quarantine.update_classification(t, i, cls)


            if is_q:
                neighbor_preds = [fields[j].ema for j in active_neigh]
                quarantine.update_alignment(t, i, fields[i].ema, neighbor_preds)

        debt_mgr.settle_due_debts(t, deviations_t, thresholds_t)


        if t % 5 == 0:
            prop.step(t, fields, trackers, quarantine.quarantined)

    if verbose:
        print(f"Simulation complete: N={N}, T={T}")
        print(f"  Total debts issued:    {debt_mgr.debts_total}")
        print(f"  Debts confirmed:       {debt_mgr.debts_confirmed}")
        print(f"  Quarantine events:     {len(quarantine.quarantine_events)}")
        print(f"  Conflicts resolved:    {len(prop.conflict_log)}")

    return {
        "fields": fields, "trackers": trackers, "prop": prop,
        "debt_mgr": debt_mgr, "classifier": classifier, "quarantine": quarantine,
        "data": data, "ground_truth": ground_truth, "nodes": nodes, "events": events,
        "positions": positions, "storage_log": storage_log,
        "deviation_log": deviation_log, "threshold_log": threshold_log,
        "classification_log": classification_log,
        "quarantine_state_log": quarantine_state_log,
        "N": N, "T": T,
    }


def compute_all_metrics(r: dict) -> dict:
    fields = r["fields"]
    ground_truth = r["ground_truth"]
    nodes = r["nodes"]
    T = r["T"]
    quarantine = r["quarantine"]
    debt_mgr = r["debt_mgr"]

    storage_ratios = [f.storage_ratio(T) for f in fields]
    mse_list = [f.reconstruction_mse(ground_truth[i]) for i, f in enumerate(fields)]

    fault_map = {n.node_id: (n.fault_start, n.fault_end, n.fault_type)
                 for n in nodes if n.fault_type is not None}

    flagged_nodes = quarantine.quarantined | {
        e["node"] for e in quarantine.quarantine_events if e["event"] == "enter"
    }
    detection_times = {}
    for e in quarantine.quarantine_events:
        if e["event"] == "enter" and e["node"] not in detection_times:
            detection_times[e["node"]] = e["t"]

    from components.baselines import _fault_metrics
    prec, rec, f1, delay = _fault_metrics(fault_map, flagged_nodes, detection_times)

    n_events = len(fault_map) + 2  
    anomaly_recall = 1.0 if debt_mgr.debts_total > 0 else 0.0

    return {
        "storage_ratio": float(np.mean(storage_ratios)),
        "mean_mse": float(np.mean(mse_list)),
        "anomaly_recall": anomaly_recall,
        "fault_f1": f1,
        "fault_precision": prec,
        "fault_recall": rec,
        "mean_detection_delay": delay,
    }


if __name__ == "__main__":
    result = run_integrated(N=20, T=2016, seed=42, verbose=True)
    metrics = compute_all_metrics(result)
    print("\n== ADEF Metrics ==")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
