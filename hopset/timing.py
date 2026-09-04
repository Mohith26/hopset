"""Net timing: how long two radios stay usefully synchronised, and how long a
late joiner takes to find the net.

Both questions have closed form answers, and both are simulated here so the
simulation can be checked against them rather than trusted.

Clock drift
-----------
Two radios agree on the hop sequence only if they agree on time of day. A
crystal specified at some parts per million accumulates error linearly, so a
radio with a 2 ppm oscillator drifts 2 microseconds per second, and the time
until it slides outside a guard interval is simply guard / (ppm * 1e-6). That
is why hopping nets resynchronise on a schedule instead of when something
breaks.

Acquisition
-----------
A joiner that does not know the net's time of day has to search. That is a
serial search over uncertainty cells with a per dwell detection probability, a
false alarm probability, and a penalty for chasing a false alarm, and it has a
standard mean acquisition time:

    E[T] = ((2 - Pd)(N - 1)(1 + K Pfa) / (2 Pd)) * Td  +  Td / Pd

which is the classic result for serial search acquisition of a spread spectrum
signal. `simulate_acquisition` reproduces it by brute force.
"""

import numpy as np


def drift_us_per_second(ppm):
    return ppm


def time_to_exceed_guard_s(ppm, guard_us):
    """Deterministic drift only. Linear, so this is exact."""
    if ppm <= 0:
        return float("inf")
    return guard_us / float(ppm)


def simulate_drift(ppm, guard_us, jitter_us_rms, rng, max_seconds=100000, trials=200):
    """Deterministic drift plus per second Gaussian jitter, to see how much the
    random component moves the crossing time away from the closed form."""
    out = []
    for _ in range(trials):
        err = 0.0
        for t in range(1, max_seconds + 1):
            err += ppm + rng.normal(0.0, jitter_us_rms)
            if abs(err) > guard_us:
                out.append(t)
                break
        else:
            out.append(max_seconds)
    arr = np.array(out, dtype=float)
    return {
        "closed_form_seconds": time_to_exceed_guard_s(ppm, guard_us),
        "mean_seconds": float(arr.mean()),
        "p05_seconds": float(np.percentile(arr, 5)),
        "p95_seconds": float(np.percentile(arr, 95)),
        "trials": trials,
    }


def resync_interval_s(ppm, guard_us, margin=0.5):
    """Resynchronise at a fraction of the time it takes to fall out, so a
    single missed resync is survivable."""
    return margin * time_to_exceed_guard_s(ppm, guard_us)


def mean_acquisition_time_closed_form(n_cells, pd, pfa, penalty_dwells, dwell_s):
    a = (2.0 - pd) * (n_cells - 1) * (1.0 + penalty_dwells * pfa) / (2.0 * pd)
    return (a + 1.0 / pd) * dwell_s


def simulate_acquisition(n_cells, pd, pfa, penalty_dwells, dwell_s, rng, trials=4000):
    """Serial search: step through uncertainty cells in order from a random
    start, dwell on each, and stop when the correct cell is both reached and
    detected. A false alarm on a wrong cell costs `penalty_dwells` before the
    search resumes."""
    times = np.empty(trials, dtype=float)
    for t in range(trials):
        truth = int(rng.integers(0, n_cells))
        cell = int(rng.integers(0, n_cells))
        dwells = 0.0
        while True:
            dwells += 1.0
            if cell == truth:
                if rng.random() < pd:
                    break
            else:
                if rng.random() < pfa:
                    dwells += penalty_dwells
            cell = (cell + 1) % n_cells
        times[t] = dwells * dwell_s
    return {
        "mean_seconds": float(times.mean()),
        "p50_seconds": float(np.percentile(times, 50)),
        "p95_seconds": float(np.percentile(times, 95)),
        "trials": trials,
    }


def uncertainty_cells(clock_error_s, hop_rate_hz):
    """One cell per hop of time of day uncertainty, both directions."""
    return int(2 * np.ceil(clock_error_s * hop_rate_hz)) + 1
