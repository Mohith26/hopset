"""End to end: bits in, bits out, across a hopping link with a jammer.

Chain: information bits, rate 1/2 K=7 convolutional encoding, interleaving
across hops, BPSK, hop assignment, channel, matched filter, log likelihood
ratios, deinterleaving, soft Viterbi.

The interleaver is the whole reason this file exists as a separate experiment.
A convolutional code is good at scattered errors and bad at consecutive ones,
and a hopping radio produces the worst possible case: when a hop lands on a
jammed channel, every bit sent during that 9 millisecond dwell dies at once.
Without interleaving a single bad hop hands the decoder a burst longer than
its memory. With interleaving across hops the same bad hop becomes isolated
errors sprinkled through the codeword, which is the shape the code was
designed for.

Four receivers are compared, because the first one I wrote produced a result I
did not believe and then turned out to be right.

  nominal    computes log likelihood ratios with the nominal noise variance,
             which is what you get if nobody thinks about jamming
  clipped    the same, with the ratio magnitude limited
  estimated  estimates the noise variance separately on each hop, blind, from
             the quadrature arm, which carries no signal under BPSK and is
             therefore a clean noise sample
  oracle     told exactly which hops were jammed and erases them, which is the
             best any jam state estimator could do

The gap between nominal and oracle is not small, and it runs the wrong way for
the whole premise of frequency hopping. See the README.
"""

import numpy as np

from . import fec
from .hop import DWELL_S, Hopset, jammed_set
from .phy import awgn, modulate, soft_llr


class FrameConfig:
    def __init__(self, hops=32, bits_per_hop=32):
        self.hops = hops
        self.bits_per_hop = bits_per_hop
        self.coded_bits = hops * bits_per_hop
        # rate 1/2 with 6 tail bits flushed
        self.info_bits = self.coded_bits // 2 - (fec.K - 1)

    @property
    def dwell_bits(self):
        return self.bits_per_hop

    @property
    def frame_seconds(self):
        return self.hops * DWELL_S


def interleave_across_hops(coded, cfg):
    """Coded bit c is transmitted on hop (c mod hops), so consecutive coded
    bits never share a dwell."""
    return coded.reshape(cfg.bits_per_hop, cfg.hops).T.reshape(-1).copy()


def deinterleave_across_hops(received, cfg):
    return received.reshape(cfg.hops, cfg.bits_per_hop).T.reshape(-1).copy()


def _llrs_for_frame(rx, cfg, nominal_var, mode, hop_is_jammed, clip=4.0):
    """Turn received symbols into soft information under one of four receiver
    models. Only `oracle` uses knowledge the receiver could not have."""
    if mode == "oracle":
        llrs = soft_llr(rx, "bpsk", nominal_var)
        for h in range(cfg.hops):
            if hop_is_jammed[h]:
                llrs[h * cfg.bits_per_hop : (h + 1) * cfg.bits_per_hop] = 0.0
        return llrs
    if mode == "estimated":
        llrs = np.empty(len(rx), dtype=float)
        for h in range(cfg.hops):
            lo, hi = h * cfg.bits_per_hop, (h + 1) * cfg.bits_per_hop
            # BPSK puts all the signal on the in phase arm, so the quadrature
            # arm of this hop is a pure noise sample and its mean square is an
            # unbiased estimate of the per dimension noise variance. No side
            # information, no training sequence, just the samples in hand.
            var_hat = max(float(np.mean(rx[lo:hi].imag ** 2)), 1e-6)
            llrs[lo:hi] = soft_llr(rx[lo:hi], "bpsk", var_hat)
        return llrs
    llrs = soft_llr(rx, "bpsk", nominal_var)
    if mode == "clipped":
        np.clip(llrs, -clip, clip, out=llrs)
    return llrs


def transmit_frame(info_bits, cfg, hopset, tod, ebn0_db, jammed, rng,
                   interleaved=True, jam_ebn0_db=-6.0, receiver="nominal"):
    """One frame. Returns the decoded information bits.

    Jamming is modelled as partial band noise: a hop that lands on a jammed
    channel is received at `jam_ebn0_db` instead of `ebn0_db`.
    """
    coded = fec.encode(info_bits, terminate=True)
    assert len(coded) == cfg.coded_bits, (len(coded), cfg.coded_bits)
    tx_bits = interleave_across_hops(coded, cfg) if interleaved else coded.copy()

    channels = hopset.sequence(tod, cfg.hops)
    hop_is_jammed = np.array([int(c) in jammed for c in channels], dtype=bool)

    symbols = modulate(tx_bits, "bpsk")
    nominal_var = None
    rx = np.empty_like(symbols)
    for h in range(cfg.hops):
        lo = h * cfg.bits_per_hop
        hi = lo + cfg.bits_per_hop
        eb = jam_ebn0_db if hop_is_jammed[h] else ebn0_db
        chunk, var = awgn(symbols[lo:hi], eb, 1, rng, code_rate=fec.RATE)
        rx[lo:hi] = chunk
        if not hop_is_jammed[h]:
            nominal_var = var
    if nominal_var is None:
        _, nominal_var = awgn(symbols[:1], ebn0_db, 1, rng, code_rate=fec.RATE)

    llrs = _llrs_for_frame(rx, cfg, nominal_var, receiver, hop_is_jammed)

    coded_llrs = deinterleave_across_hops(llrs, cfg) if interleaved else llrs
    return fec.decode(coded_llrs, cfg.info_bits, terminated=True), int(hop_is_jammed.sum())


def frame_error_rate(cfg, ebn0_db, jam_fraction, trials, seed,
                     interleaved=True, jammer="partial_band",
                     receiver="nominal"):
    rng = np.random.default_rng(seed)
    hopset = Hopset(key=b"net-key-for-simulation-only", net_id=0x1234)
    errors = 0
    bit_errors = 0
    total_bits = 0
    jammed_hops = 0
    for t in range(trials):
        jam = jammed_set(jammer, jam_fraction, rng)
        info = rng.integers(0, 2, cfg.info_bits).astype(np.int8)
        out, nj = transmit_frame(
            info, cfg, hopset, tod=1000 * t, ebn0_db=ebn0_db, jammed=jam, rng=rng,
            interleaved=interleaved, receiver=receiver,
        )
        jammed_hops += nj
        diff = int(np.sum(out != info))
        bit_errors += diff
        total_bits += cfg.info_bits
        if diff:
            errors += 1
    return {
        "frame_error_rate": errors / float(trials),
        "bit_error_rate": bit_errors / float(total_bits),
        "mean_jammed_hops_per_frame": jammed_hops / float(trials),
        "trials": trials,
    }


def fixed_frequency_frame_error_rate(cfg, ebn0_db, jam_fraction, trials, seed,
                                     jammer="partial_band", jam_ebn0_db=-6.0):
    """The same frame sent on one channel for its whole duration, which is what
    a non hopping radio does. Either the channel is clean and everything gets
    through, or it is jammed and nothing does."""
    rng = np.random.default_rng(seed)
    errors = 0
    for t in range(trials):
        jam = jammed_set(jammer, jam_fraction, rng)
        channel = int(rng.integers(0, 2320))
        info = rng.integers(0, 2, cfg.info_bits).astype(np.int8)
        coded = fec.encode(info, terminate=True)
        eb = jam_ebn0_db if channel in jam else ebn0_db
        symbols = modulate(coded, "bpsk")
        rx, var = awgn(symbols, eb, 1, rng, code_rate=fec.RATE)
        llrs = soft_llr(rx, "bpsk", var if channel not in jam else var)
        out = fec.decode(llrs, cfg.info_bits, terminated=True)
        if int(np.sum(out != info)):
            errors += 1
    return {"frame_error_rate": errors / float(trials), "trials": trials}
