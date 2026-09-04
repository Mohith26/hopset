"""Every number in the README comes from here.

    python -m hopset.experiments ber
    python -m hopset.experiments coding_gain
    python -m hopset.experiments scheduling
    python -m hopset.experiments inversion
    python -m hopset.experiments jamming
    python -m hopset.experiments acquisition
    python -m hopset.experiments combine
"""

import json
import os
import sys
import time

import numpy as np

from . import hop, timing
from .fec import RATE, decode, encode
from .phy import awgn, demodulate, modulate, soft_llr, theoretical_ber_bpsk
from .rt import (
    Task,
    assign_rate_monotonic,
    hyperperiod,
    liu_layland_bound,
    response_time_analysis,
    simulate,
    radio_task_set,
    total_utilization,
)
from .waveform import FrameConfig, fixed_frequency_frame_error_rate, frame_error_rate

OUT_DIR = "results"


def _save(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "part_%s.json" % name), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def ber(seed=101, n_bits=2_000_000):
    """Simulated bit error rate against the closed form, which is the only
    check that says the whole baseband chain is calibrated rather than merely
    self consistent."""
    rng = np.random.default_rng(seed)
    rows = []
    for scheme, bps in (("bpsk", 1), ("qpsk", 2)):
        for ebn0 in (0.0, 2.0, 4.0, 6.0, 8.0):
            bits = rng.integers(0, 2, n_bits).astype(np.int8)
            sym = modulate(bits, scheme)
            rx, _ = awgn(sym, ebn0, bps, rng)
            measured = float(np.mean(demodulate(rx, scheme) != bits))
            theory = float(theoretical_ber_bpsk(ebn0))
            rows.append({
                "scheme": scheme,
                "ebn0_db": ebn0,
                "measured_ber": measured,
                "theoretical_ber": theory,
                "ratio": round(measured / theory, 4) if theory else None,
                "bits": n_bits,
            })
    worst = max(abs(r["ratio"] - 1.0) for r in rows if r["ratio"])
    return {"rows": rows, "worst_relative_deviation": round(worst, 4), "seed": seed}


def _ebn0_at_ber(points, target=1e-3):
    """Log linear interpolation of the Eb/N0 that hits a target error rate."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= target >= y1 and y0 > 0 and y1 > 0:
            lx0, lx1 = np.log10(y0), np.log10(y1)
            t = (np.log10(target) - lx0) / (lx1 - lx0)
            return float(x0 + t * (x1 - x0))
    return None


def coding_gain(seed=103, info_bits=40000):
    """What the convolutional code actually buys, measured rather than quoted.

    Both coded curves pay for the code: the noise is scaled by the code rate,
    so the x axis is energy per information bit in every case. Dropping that
    term is the classic way to publish a coding gain that does not exist.
    """
    rng = np.random.default_rng(seed)
    uncoded, hard, soft = [], [], []
    n_unc = 1_000_000
    for ebn0 in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        bits = rng.integers(0, 2, n_unc).astype(np.int8)
        rx, _ = awgn(modulate(bits, "bpsk"), ebn0, 1, rng)
        uncoded.append((ebn0, float(np.mean(demodulate(rx, "bpsk") != bits))))
    for ebn0 in (1.0, 2.0, 3.0, 4.0, 5.0):
        bits = rng.integers(0, 2, info_bits).astype(np.int8)
        coded = encode(bits)
        rx, var = awgn(modulate(coded, "bpsk"), ebn0, 1, rng, code_rate=RATE)
        s = decode(soft_llr(rx, "bpsk", var), info_bits)
        h = decode(2.0 * demodulate(rx, "bpsk").astype(float) - 1.0, info_bits)
        soft.append((ebn0, float(np.mean(s != bits))))
        hard.append((ebn0, float(np.mean(h != bits))))
    target = 1e-3
    e_unc = _ebn0_at_ber(uncoded, target)
    e_hard = _ebn0_at_ber(hard, target)
    e_soft = _ebn0_at_ber(soft, target)
    return {
        "seed": seed,
        "target_ber": target,
        "uncoded": [{"ebn0_db": a, "ber": b} for a, b in uncoded],
        "coded_hard_decision": [{"ebn0_db": a, "ber": b} for a, b in hard],
        "coded_soft_decision": [{"ebn0_db": a, "ber": b} for a, b in soft],
        "ebn0_for_target_uncoded_db": round(e_unc, 3) if e_unc else None,
        "ebn0_for_target_hard_db": round(e_hard, 3) if e_hard else None,
        "ebn0_for_target_soft_db": round(e_soft, 3) if e_soft else None,
        "coding_gain_soft_db": round(e_unc - e_soft, 2) if e_unc and e_soft else None,
        "coding_gain_hard_db": round(e_unc - e_hard, 2) if e_unc and e_hard else None,
        "soft_over_hard_db": round(e_hard - e_soft, 2) if e_hard and e_soft else None,
        "info_bits_per_coded_point": info_bits,
    }


def scheduling():
    """Response time analysis against the simulator, for the radio task set and
    for a textbook set that fails the utilisation test but is schedulable."""
    out = {}
    for name, tasks, horizon_mult in (
        ("radio_task_set", radio_task_set(), 3),
        ("above_utilization_bound", [Task("t1", 7, 3), Task("t2", 12, 3), Task("t3", 20, 5)], 40),
    ):
        assign_rate_monotonic(tasks)
        analysis = response_time_analysis(tasks)
        sim = simulate(tasks, hyperperiod(tasks) * horizon_mult)
        rows = []
        for t in sorted(tasks, key=lambda x: x.priority):
            rows.append({
                "task": t.name,
                "priority": t.priority,
                "period": t.period,
                "wcet": t.wcet,
                "blocking": analysis[t.name]["blocking"],
                "analytic_worst_response": analysis[t.name]["response_time"],
                "observed_worst_response": sim.worst_response[t.name],
                "deadline": t.deadline,
                "schedulable": analysis[t.name]["schedulable"],
                "slack_ticks": t.deadline - analysis[t.name]["response_time"],
            })
        u = total_utilization(tasks)
        out[name] = {
            "tasks": rows,
            "utilization": round(u, 4),
            "liu_layland_bound": round(liu_layland_bound(len(tasks)), 4),
            "passes_utilization_test": u <= liu_layland_bound(len(tasks)),
            "passes_response_time_analysis": all(r["schedulable"] for r in rows),
            "hyperperiod_ticks": hyperperiod(tasks),
            "deadline_misses": sim.deadline_misses,
            "analysis_pessimism_ticks": {
                r["task"]: r["analytic_worst_response"] - r["observed_worst_response"]
                for r in rows
            },
        }
    return out


def inversion():
    """The Mars Pathfinder shape: a high priority task blocked by a low
    priority task's critical section while an unrelated medium priority task
    runs in front of the holder."""
    def build():
        return [
            Task("high", period=100, wcet=6, cs_len=3, cs_start=1, priority=0, offset=5),
            Task("medium", period=100, wcet=30, priority=1, offset=6),
            Task("low", period=100, wcet=20, cs_len=15, cs_start=1, priority=2, offset=0),
        ]
    without = simulate(build(), 400, priority_inheritance=False)
    with_pi = simulate(build(), 400, priority_inheritance=True)
    return {
        "worst_response_high_without_inheritance": without.worst_response["high"],
        "worst_response_high_with_inheritance": with_pi.worst_response["high"],
        "worst_blocking_high_without_inheritance": without.worst_blocking["high"],
        "worst_blocking_high_with_inheritance": with_pi.worst_blocking["high"],
        "medium_task_wcet": 30,
        "low_task_critical_section": 15,
        "ticks_saved": without.worst_response["high"] - with_pi.worst_response["high"],
    }


def jamming(seed=107, trials=150):
    cfg = FrameConfig(hops=32, bits_per_hop=32)
    rows = []
    for frac in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
        il = frame_error_rate(cfg, 6.0, frac, trials, seed, interleaved=True)
        no_il = frame_error_rate(cfg, 6.0, frac, trials, seed, interleaved=False)
        clipped = frame_error_rate(cfg, 6.0, frac, trials, seed, interleaved=True,
                                   receiver="clipped")
        estimated = frame_error_rate(cfg, 6.0, frac, trials, seed, interleaved=True,
                                     receiver="estimated")
        erasure = frame_error_rate(cfg, 6.0, frac, trials, seed, interleaved=True,
                                   receiver="oracle")
        fixed = fixed_frequency_frame_error_rate(cfg, 6.0, frac, trials, seed)
        rows.append({
            "jammed_fraction": frac,
            "fer_hopping_interleaved": il["frame_error_rate"],
            "fer_hopping_no_interleaving": no_il["frame_error_rate"],
            "fer_hopping_llr_clipped": clipped["frame_error_rate"],
            "fer_hopping_per_hop_variance_estimate": estimated["frame_error_rate"],
            "fer_hopping_with_oracle_jam_state": erasure["frame_error_rate"],
            "fer_fixed_frequency": fixed["frame_error_rate"],
            "mean_jammed_hops_per_frame": round(il["mean_jammed_hops_per_frame"], 2),
        })
    crossovers = [r["jammed_fraction"] for r in rows
                  if r["fer_hopping_interleaved"] > r["fer_fixed_frequency"]]
    blind_crossovers = [r["jammed_fraction"] for r in rows
                        if r["fer_hopping_per_hop_variance_estimate"] > r["fer_fixed_frequency"]]
    return {
        "seed": seed,
        "trials_per_point": trials,
        "ebn0_db": 6.0,
        "hops_per_frame": cfg.hops,
        "bits_per_hop": cfg.bits_per_hop,
        "info_bits_per_frame": cfg.info_bits,
        "frame_seconds": round(cfg.frame_seconds, 4),
        "rows": rows,
        "jammed_fractions_where_fixed_frequency_beats_nominal_receiver": crossovers,
        "jammed_fractions_where_fixed_frequency_beats_blind_estimating_receiver": blind_crossovers,
        "processing_gain_db": round(hop.processing_gain_db(), 2),
        "follower_jammer_fraction_of_dwell_corrupted": {
            "1_ms_reaction": round(hop.follower_jammer_catches(0.001), 3),
            "4.5_ms_reaction": round(hop.follower_jammer_catches(0.0045), 3),
            "9_ms_reaction": round(hop.follower_jammer_catches(0.009), 3),
            "10_ms_reaction": round(hop.follower_jammer_catches(0.010), 3),
        },
    }


def acquisition(seed=109):
    rng = np.random.default_rng(seed)
    rows = []
    for clock_error_s in (0.05, 0.27, 1.0, 5.0):
        cells = timing.uncertainty_cells(clock_error_s, hop.HOP_RATE_HZ)
        closed = timing.mean_acquisition_time_closed_form(cells, 0.9, 0.01, 20, hop.DWELL_S)
        sim = timing.simulate_acquisition(cells, 0.9, 0.01, 20, hop.DWELL_S, rng, trials=3000)
        rows.append({
            "clock_error_s": clock_error_s,
            "uncertainty_cells": cells,
            "closed_form_mean_s": round(closed, 4),
            "simulated_mean_s": round(sim["mean_seconds"], 4),
            "simulated_p95_s": round(sim["p95_seconds"], 4),
            "relative_error": round(abs(sim["mean_seconds"] - closed) / closed, 4),
        })
    drift = {}
    for ppm in (0.5, 2.0, 10.0):
        for guard_us in (200.0, 500.0):
            drift["%.1fppm_%dus_guard" % (ppm, guard_us)] = {
                "seconds_to_fall_out_of_guard": round(
                    timing.time_to_exceed_guard_s(ppm, guard_us), 1),
                "recommended_resync_interval_s": round(
                    timing.resync_interval_s(ppm, guard_us), 1),
            }
    return {"seed": seed, "acquisition": rows, "clock_drift": drift}


SECTIONS = {
    "ber": ber,
    "coding_gain": coding_gain,
    "scheduling": scheduling,
    "inversion": inversion,
    "jamming": jamming,
    "acquisition": acquisition,
}


def combine():
    out = {}
    for name in SECTIONS:
        path = os.path.join(OUT_DIR, "part_%s.json" % name)
        out[name] = json.load(open(path)) if os.path.exists(path) else None
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s/results.json" % OUT_DIR)


def main():
    if len(sys.argv) < 2:
        print("sections: %s, combine" % ", ".join(sorted(SECTIONS)))
        return
    name = sys.argv[1]
    if name == "combine":
        combine()
        return
    t0 = time.time()
    _save(name, SECTIONS[name]())
    print("(%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
