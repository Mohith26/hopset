"""Test suite. `python -m hopset.test_hopset`.

Almost everything here is checked against something written by somebody else:
the Q function against tabulated values, the simulated bit error rate against
the closed form for coherent BPSK over AWGN, the response time analysis against
a worked example out of the real time scheduling literature, and the
acquisition simulator against the standard serial search mean.

That matters more than usual in this kind of code, because a DSP chain that is
subtly wrong still runs, still produces a curve, and the curve still bends the
right way. It is just in the wrong place, and nothing tells you.
"""

import math
import sys
import time

import numpy as np

from . import fec, hop, timing
from .phy import (
    awgn,
    demodulate,
    modulate,
    q,
    rrc_isi_check,
    rrc_taps,
    shape_and_match,
    soft_llr,
    theoretical_ber_bpsk,
)
from .rt import (
    Task,
    assign_rate_monotonic,
    blocking_time,
    hyperperiod,
    liu_layland_bound,
    radio_task_set,
    response_time_analysis,
    simulate,
    total_utilization,
)
from .waveform import FrameConfig, deinterleave_across_hops, frame_error_rate, interleave_across_hops

_T0 = time.time()
R = {"passed": 0, "assertions": 0, "failed": 0, "failures": []}


def check(name, cond, detail=""):
    R["assertions"] += 1
    if not cond:
        R["failed"] += 1
        R["failures"].append("%s: %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))
    return cond


def case(fn):
    before = R["failed"]
    fn()
    if R["failed"] == before:
        R["passed"] += 1
        print("  ok   %s" % fn.__name__)
    return fn


# --------------------------------------------------------------------- phy


@case
def test_q_function_against_tables():
    for x, want in [(0.0, 0.5), (1.0, 0.158655), (2.0, 0.0227501), (3.0, 0.00134990),
                    (4.0, 3.16712e-5)]:
        got = float(q(x))
        check("Q(%.1f)" % x, abs(got - want) / want < 1e-5, "%g vs %g" % (got, want))


@case
def test_rrc_is_nyquist():
    check("taps are unit energy", abs(np.sum(rrc_taps(8, 10, 0.35) ** 2) - 1.0) < 1e-9)
    for beta in (0.2, 0.35, 0.5):
        isi = rrc_isi_check(8, 12, beta)
        check("peak ISI at beta=%.2f is negligible" % beta, isi < 5e-3, "%.2e" % isi)
    # the singular tap at t = 1/(4 beta) must be finite
    taps = rrc_taps(4, 10, 0.25)
    check("no NaN or inf in taps", np.all(np.isfinite(taps)))


@case
def test_shaped_chain_matches_ideal_symbols():
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, 2000).astype(np.int8)
    symbols = modulate(bits, "bpsk")
    recovered = shape_and_match(symbols, sps=8, span=12, beta=0.35)
    n = min(len(symbols), len(recovered))
    err = np.max(np.abs(recovered[20:n - 20] - symbols[20:n - 20]))
    check("matched filter recovers the symbols", err < 1e-2, "max err %.2e" % err)
    print("       matched filter residual, truncated to 12 symbols: %.2e" % err)


@case
def test_ber_matches_theory():
    rng = np.random.default_rng(7)
    n = 200000
    for scheme, bits_per_symbol in (("bpsk", 1), ("qpsk", 2)):
        for ebn0 in (2.0, 4.0, 6.0):
            bits = rng.integers(0, 2, n).astype(np.int8)
            symbols = modulate(bits, scheme)
            rx, _ = awgn(symbols, ebn0, bits_per_symbol, rng)
            out = demodulate(rx, scheme)
            measured = float(np.mean(out != bits))
            want = float(theoretical_ber_bpsk(ebn0))
            check(
                "%s BER at %.0f dB" % (scheme, ebn0),
                abs(measured - want) / want < 0.15,
                "measured %.5f vs theory %.5f" % (measured, want),
            )


@case
def test_awgn_accounts_for_code_rate():
    """Forgetting the code rate makes a rate 1/2 code look 3 dB better than it
    is, so this pins the noise variance directly."""
    rng = np.random.default_rng(9)
    sym = modulate(np.zeros(200000, dtype=np.int8), "bpsk")
    _, var_uncoded = awgn(sym, 5.0, 1, rng, code_rate=1.0)
    _, var_coded = awgn(sym, 5.0, 1, rng, code_rate=0.5)
    check("halving the rate doubles the noise variance",
          abs(var_coded / var_uncoded - 2.0) < 1e-9, "%.4f" % (var_coded / var_uncoded))


# --------------------------------------------------------------------- fec


@case
def test_trellis_structure():
    check("64 states", fec.N_STATES == 64)
    for s in range(fec.N_STATES):
        for b in (0, 1):
            ns = fec.NEXT_STATE[s, b]
            check("transition stays in range", 0 <= ns < 64)
    # every state must be reachable from exactly two predecessors
    counts = np.zeros(64, dtype=int)
    for s in range(64):
        for b in (0, 1):
            counts[fec.NEXT_STATE[s, b]] += 1
    check("every state has two predecessors", np.all(counts == 2))
    check("generators are the standard pair", fec.G == (0o171, 0o133))


@case
def test_noiseless_roundtrip_is_exact():
    rng = np.random.default_rng(11)
    for n in (16, 100, 501, 2000):
        bits = rng.integers(0, 2, n).astype(np.int8)
        coded = fec.encode(bits)
        check("rate 1/2 with tail", len(coded) == 2 * (n + fec.K - 1))
        llrs = 2.0 * coded.astype(float) - 1.0
        out = fec.decode(llrs, n)
        check("exact recovery at n=%d" % n, np.array_equal(out, bits),
              "%d differing" % int(np.sum(out != bits)))


@case
def test_decoder_corrects_isolated_errors():
    rng = np.random.default_rng(13)
    bits = rng.integers(0, 2, 400).astype(np.int8)
    coded = fec.encode(bits)
    llrs = 2.0 * coded.astype(float) - 1.0
    flips = rng.choice(len(llrs), size=12, replace=False)
    llrs[flips] *= -1.0
    out = fec.decode(llrs, 400)
    check("12 scattered hard errors are corrected", np.array_equal(out, bits),
          "%d bits wrong" % int(np.sum(out != bits)))


@case
def test_interleaver_is_a_permutation():
    cfg = FrameConfig(hops=16, bits_per_hop=8)
    x = np.arange(cfg.coded_bits)
    y = interleave_across_hops(x, cfg)
    check("permutation", sorted(y.tolist()) == list(range(cfg.coded_bits)))
    check("inverts", np.array_equal(deinterleave_across_hops(y, cfg), x))
    spread = [int(np.where(y == c)[0][0]) // cfg.bits_per_hop for c in range(cfg.hops)]
    check("consecutive coded bits land on distinct hops",
          len(set(spread)) == cfg.hops, str(spread[:8]))
    block = fec.BlockInterleaver(8, 4)
    z = np.arange(32)
    check("block interleaver inverts", np.array_equal(block.deinterleave(block.interleave(z)), z))


@case
def test_soft_decision_beats_hard_decision():
    """Textbook result: soft decision Viterbi is worth roughly 2 dB over hard
    decision on AWGN. If the LLR scaling is wrong this collapses."""
    rng = np.random.default_rng(17)
    n = 8000
    ebn0 = 3.0
    bits = rng.integers(0, 2, n).astype(np.int8)
    coded = fec.encode(bits)
    symbols = modulate(coded, "bpsk")
    rx, var = awgn(symbols, ebn0, 1, rng, code_rate=fec.RATE)
    soft = fec.decode(soft_llr(rx, "bpsk", var), n)
    hard_bits = demodulate(rx, "bpsk")
    hard = fec.decode(2.0 * hard_bits.astype(float) - 1.0, n)
    ber_soft = float(np.mean(soft != bits))
    ber_hard = float(np.mean(hard != bits))
    check("soft decoding is better than hard", ber_soft < ber_hard,
          "soft %.5f hard %.5f" % (ber_soft, ber_hard))
    print("       coded BER at 3 dB: soft %.5f, hard %.5f" % (ber_soft, ber_hard))


# ---------------------------------------------------------------------- rt


@case
def test_liu_layland_values():
    check("n=1", abs(liu_layland_bound(1) - 1.0) < 1e-12)
    check("n=2", abs(liu_layland_bound(2) - 0.8284271) < 1e-6)
    check("n=3", abs(liu_layland_bound(3) - 0.7797631) < 1e-6)
    check("converges to ln2", abs(liu_layland_bound(100000) - math.log(2)) < 1e-4)


@case
def test_rta_matches_a_worked_example():
    """A standard three task example: utilisation 0.929, far above the
    Liu and Layland bound of 0.780, and still schedulable, with the lowest
    priority task finishing exactly on its deadline."""
    tasks = [Task("t1", 7, 3), Task("t2", 12, 3), Task("t3", 20, 5)]
    assign_rate_monotonic(tasks)
    u = total_utilization(tasks)
    check("utilisation", abs(u - (3 / 7 + 3 / 12 + 5 / 20)) < 1e-12)
    check("above the sufficient bound", u > liu_layland_bound(3))
    r = response_time_analysis(tasks)
    check("R1", r["t1"]["response_time"] == 3, str(r["t1"]))
    check("R2", r["t2"]["response_time"] == 6, str(r["t2"]))
    check("R3", r["t3"]["response_time"] == 20, str(r["t3"]))
    check("all schedulable", all(v["schedulable"] for v in r.values()))


@case
def test_simulation_never_exceeds_the_analytic_bound():
    tasks = [Task("t1", 7, 3), Task("t2", 12, 3), Task("t3", 20, 5)]
    assign_rate_monotonic(tasks)
    r = response_time_analysis(tasks)
    sim = simulate(tasks, hyperperiod(tasks) * 3)
    for t in tasks:
        check("%s observed <= analytic" % t.name,
              sim.worst_response[t.name] <= r[t.name]["response_time"],
              "observed %d, bound %d" % (sim.worst_response[t.name], r[t.name]["response_time"]))
    check("no deadline misses", not sim.any_miss, str(sim.deadline_misses))
    check("critical instant reaches the bound for the lowest priority task",
          sim.worst_response["t3"] == r["t3"]["response_time"],
          "%d vs %d" % (sim.worst_response["t3"], r["t3"]["response_time"]))


@case
def test_radio_task_set_is_schedulable_and_simulation_agrees():
    tasks = radio_task_set()
    assign_rate_monotonic(tasks)
    r = response_time_analysis(tasks)
    check("hop timer meets its dwell", r["hop_timer"]["schedulable"], str(r["hop_timer"]))
    sim = simulate(tasks, hyperperiod(tasks) * 2)
    for t in tasks:
        check("%s observed <= analytic" % t.name,
              sim.worst_response[t.name] <= r[t.name]["response_time"],
              "observed %d bound %d" % (sim.worst_response[t.name], r[t.name]["response_time"]))
    check("every task completed at least once",
          all(v > 0 for v in sim.completions.values()), str(sim.completions))


@case
def test_priority_inversion_and_inheritance():
    """A high priority task blocked by a long critical section in a low
    priority task, with a medium priority task in between that does not touch
    the resource. Without inheritance the medium task preempts the resource
    holder and the high priority task waits for both. With inheritance the
    holder runs at the blocked task's priority and the medium task cannot cut
    in front."""
    tasks = [
        Task("high", period=100, wcet=6, cs_len=3, cs_start=1, priority=0, offset=5),
        Task("medium", period=100, wcet=30, priority=1, offset=6),
        Task("low", period=100, wcet=20, cs_len=15, cs_start=1, priority=2, offset=0),
    ]
    # low starts first and is inside its critical section when high arrives
    without = simulate(tasks, 400, priority_inheritance=False)
    with_pi = simulate(tasks, 400, priority_inheritance=True)
    b = blocking_time(tasks, tasks[0])
    check("analytic blocking is the long critical section", b == 15, str(b))
    check("inheritance cuts the high priority task's worst response",
          with_pi.worst_response["high"] < without.worst_response["high"],
          "with %d without %d" % (with_pi.worst_response["high"],
                                  without.worst_response["high"]))
    check("the unbounded case is at least the medium task long",
          without.worst_response["high"] > tasks[1].wcet, str(without.worst_response))
    print("       high priority worst response: %d ticks without inheritance, %d with"
          % (without.worst_response["high"], with_pi.worst_response["high"]))


# --------------------------------------------------------------------- hop


@case
def test_band_constants():
    check("2320 channels spans the band",
          hop.channel_hz(hop.NUM_CHANNELS - 1) == 87_975_000,
          str(hop.channel_hz(hop.NUM_CHANNELS - 1)))
    check("first channel", hop.channel_hz(0) == 30_000_000)
    pg = hop.processing_gain_db()
    check("processing gain near 33.6 dB", 33.0 < pg < 34.0, "%.2f" % pg)
    check("dwell is about 9 ms", abs(hop.DWELL_S - 0.009009) < 1e-5)


@case
def test_hopset_is_deterministic_and_key_dependent():
    a = hop.Hopset(b"key-a", 0x1234)
    b = hop.Hopset(b"key-a", 0x1234)
    c = hop.Hopset(b"key-b", 0x1234)
    d = hop.Hopset(b"key-a", 0x9999)
    s1 = a.sequence(500, 400)
    check("same inputs give the same sequence", np.array_equal(s1, b.sequence(500, 400)))
    check("a different key gives a different sequence",
          not np.array_equal(s1, c.sequence(500, 400)))
    check("a different net gives a different sequence",
          not np.array_equal(s1, d.sequence(500, 400)))
    check("time of day advances the sequence",
          np.array_equal(a.sequence(501, 399), s1[1:]))


@case
def test_lockouts_are_respected_without_biasing():
    locked = set(range(0, 500))
    hs = hop.Hopset(b"key", 1, lockout=locked)
    seq = hs.sequence(0, 20000)
    check("never lands on a locked channel", not any(int(c) in locked for c in seq))
    chi, dof = hop.uniformity_chi_square(seq, lockout=locked)
    # a redraw keeps the distribution uniform over the allowed set; a modulo
    # remap would pile the excluded mass onto low indices and blow this up
    check("still uniform over allowed channels", chi < dof + 6 * math.sqrt(2 * dof),
          "chi2 %.1f dof %d" % (chi, dof))


@case
def test_follower_jammer_needs_to_be_faster_than_the_dwell():
    check("slow jammer accomplishes nothing", hop.follower_jammer_catches(0.010) == 0.0)
    frac = hop.follower_jammer_catches(0.0045)
    check("a jammer at half the dwell corrupts about half of it",
          abs(frac - 0.5005) < 0.01, "%.4f" % frac)


# ------------------------------------------------------------------ timing


@case
def test_clock_drift_closed_form():
    check("2 ppm against a 500 us guard",
          abs(timing.time_to_exceed_guard_s(2.0, 500.0) - 250.0) < 1e-9)
    rng = np.random.default_rng(23)
    sim = timing.simulate_drift(2.0, 500.0, jitter_us_rms=0.4, rng=rng, trials=120)
    rel = abs(sim["mean_seconds"] - sim["closed_form_seconds"]) / sim["closed_form_seconds"]
    check("simulated drift matches the closed form", rel < 0.05,
          "%.1f vs %.1f" % (sim["mean_seconds"], sim["closed_form_seconds"]))


@case
def test_acquisition_matches_serial_search_theory():
    rng = np.random.default_rng(29)
    n_cells, pd, pfa, penalty = 61, 0.9, 0.01, 20
    closed = timing.mean_acquisition_time_closed_form(n_cells, pd, pfa, penalty, hop.DWELL_S)
    sim = timing.simulate_acquisition(n_cells, pd, pfa, penalty, hop.DWELL_S, rng, trials=4000)
    rel = abs(sim["mean_seconds"] - closed) / closed
    check("serial search mean matches theory", rel < 0.08,
          "sim %.4f s vs closed form %.4f s (%.1f%%)" % (sim["mean_seconds"], closed, rel * 100))
    check("uncertainty cells", timing.uncertainty_cells(0.27, 111.0) == 61,
          str(timing.uncertainty_cells(0.27, 111.0)))
    print("       acquisition: sim %.3f s, closed form %.3f s" % (sim["mean_seconds"], closed))


# ---------------------------------------------------------------- waveform


@case
def test_frame_geometry():
    cfg = FrameConfig(hops=32, bits_per_hop=32)
    check("coded bits", cfg.coded_bits == 1024)
    check("info bits leave room for the tail", cfg.info_bits == 512 - 6)
    check("frame duration", abs(cfg.frame_seconds - 32 * hop.DWELL_S) < 1e-9)


@case
def test_clean_channel_frames_get_through():
    cfg = FrameConfig(hops=16, bits_per_hop=32)
    res = frame_error_rate(cfg, ebn0_db=6.0, jam_fraction=0.0, trials=60, seed=31)
    check("no jamming and good SNR means no frame errors",
          res["frame_error_rate"] == 0.0, str(res))


@case
def test_blind_variance_estimate_beats_a_nominal_receiver():
    """The per hop estimator only looks at the quadrature arm, which carries no
    signal under BPSK, so it is blind. It should still close most of the gap to
    a receiver that is simply told which hops were jammed."""
    cfg = FrameConfig(hops=32, bits_per_hop=32)
    nominal = frame_error_rate(cfg, 6.0, 0.15, trials=100, seed=41, receiver="nominal")
    blind = frame_error_rate(cfg, 6.0, 0.15, trials=100, seed=41, receiver="estimated")
    oracle = frame_error_rate(cfg, 6.0, 0.15, trials=100, seed=41, receiver="oracle")
    check("blind estimate beats the nominal receiver",
          blind["frame_error_rate"] < nominal["frame_error_rate"],
          "blind %.3f nominal %.3f" % (blind["frame_error_rate"], nominal["frame_error_rate"]))
    check("oracle is still at least as good as blind",
          oracle["frame_error_rate"] <= blind["frame_error_rate"] + 0.05,
          "oracle %.3f blind %.3f" % (oracle["frame_error_rate"], blind["frame_error_rate"]))
    print("       15%% jammed, frame error rate: nominal %.3f, blind estimate %.3f, oracle %.3f"
          % (nominal["frame_error_rate"], blind["frame_error_rate"], oracle["frame_error_rate"]))


@case
def test_interleaving_helps_under_jamming():
    cfg = FrameConfig(hops=32, bits_per_hop=32)
    on = frame_error_rate(cfg, 6.0, 0.10, trials=120, seed=37, interleaved=True)
    off = frame_error_rate(cfg, 6.0, 0.10, trials=120, seed=37, interleaved=False)
    check("interleaved does better than not",
          on["frame_error_rate"] < off["frame_error_rate"],
          "on %.3f off %.3f" % (on["frame_error_rate"], off["frame_error_rate"]))
    print("       10%% band jammed, frame error rate: %.3f interleaved, %.3f not"
          % (on["frame_error_rate"], off["frame_error_rate"]))


def main():
    print("\n%d cases passed, %d assertions, %d failures, %.2fs"
          % (R["passed"], R["assertions"], R["failed"], time.time() - _T0))
    if R["failures"]:
        for f in R["failures"]:
            print("  - %s" % f)
        sys.exit(1)


if __name__ == "__main__":
    main()
