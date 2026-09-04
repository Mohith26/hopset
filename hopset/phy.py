"""Baseband physical layer: pulse shaping, an AWGN channel, and a matched filter.

Everything here is checked against closed form theory rather than against
itself. For coherent BPSK and QPSK over AWGN the bit error probability is
Q(sqrt(2 * Eb/N0)) in both cases, which is a hard oracle: if the simulated
curve does not sit on top of that line, something in the noise scaling, the
pulse shaping, or the decision rule is wrong, and no amount of unit testing the
pieces in isolation will tell you which.

The pulse shape is a root raised cosine, split between transmitter and
receiver so the cascade is a full raised cosine and satisfies the Nyquist
criterion, meaning zero intersymbol interference at the ideal sampling
instants. That is a second oracle available for free, and `rrc_isi_check`
measures it.
"""

import numpy as np


def _erfc(x):
    """Complementary error function without scipy.

    Numerical Recipes' Chebyshev approximation, fractional error everywhere
    below 1.2e-7, which is far tighter than the Monte Carlo noise on any BER
    estimate this project produces.
    """
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 2.0 / (2.0 + z)
    ty = 4.0 * t - 2.0
    coeffs = [
        -1.3026537197817094,
        6.4196979235649026e-1,
        1.9476473204185836e-2,
        -9.561514786808631e-3,
        -9.46595344482036e-4,
        3.66839497852761e-4,
        4.2523324806907e-5,
        -2.0278578112534e-5,
        -1.624290004647e-6,
        1.303655835580e-6,
        1.5626441722e-8,
        -8.5238095915e-8,
        6.529054439e-9,
        5.059343495e-9,
        -9.91364156e-10,
        -2.27365122e-10,
        9.6467911e-11,
        2.394038e-12,
        -6.886027e-12,
        8.94487e-13,
        3.13092e-13,
        -1.12708e-13,
        3.81e-16,
        7.106e-15,
    ]
    d = 0.0
    dd = 0.0
    for c in coeffs[:0:-1]:
        tmp = d
        d = ty * d - dd + c
        dd = tmp
    ans = t * np.exp(-z * z + 0.5 * (coeffs[0] + ty * d) - dd)
    return np.where(x >= 0.0, ans, 2.0 - ans)


def q(x):
    return 0.5 * _erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def theoretical_ber_bpsk(ebn0_db):
    """Q(sqrt(2 Eb/N0)). Same expression for QPSK with Gray mapping, because
    QPSK is two independent BPSK channels in quadrature carrying twice the bits
    in the same bandwidth."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return q(np.sqrt(2.0 * ebn0))


def rrc_taps(sps, span_symbols, beta):
    """Root raised cosine filter taps.

    The removable singularities at t = 0 and t = +/- T/(4*beta) are handled
    explicitly. Leaving them to floating point produces 0/0 and a filter that
    looks fine in a plot and quietly destroys the BER.
    """
    n = span_symbols * sps
    t = (np.arange(-n / 2, n / 2 + 1)) / float(sps)
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12:
            taps[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))


def rrc_isi_check(sps, span_symbols, beta):
    """Peak intersymbol interference of the transmit/receive cascade.

    A root raised cosine convolved with itself is a full raised cosine, which
    is Nyquist, so every sample at a non-zero multiple of the symbol period
    should be zero. Returns the largest one relative to the main lobe.
    """
    taps = rrc_taps(sps, span_symbols, beta)
    full = np.convolve(taps, taps)
    centre = len(full) // 2
    peak = full[centre]
    offsets = [
        centre + k * sps
        for k in range(-span_symbols, span_symbols + 1)
        if k != 0 and 0 <= centre + k * sps < len(full)
    ]
    return float(max(abs(full[o]) for o in offsets) / abs(peak))


def modulate(bits, scheme="bpsk"):
    """Gray mapped BPSK or QPSK symbols with unit average energy."""
    bits = np.asarray(bits, dtype=np.int8)
    if scheme == "bpsk":
        return (2.0 * bits - 1.0).astype(complex)
    if scheme == "qpsk":
        if len(bits) % 2:
            raise ValueError("QPSK needs an even number of bits")
        i = 2.0 * bits[0::2] - 1.0
        q_arm = 2.0 * bits[1::2] - 1.0
        return (i + 1j * q_arm) / np.sqrt(2.0)
    raise ValueError("unknown scheme %r" % scheme)


def demodulate(symbols, scheme="bpsk"):
    if scheme == "bpsk":
        return (symbols.real > 0).astype(np.int8)
    if scheme == "qpsk":
        out = np.empty(2 * len(symbols), dtype=np.int8)
        out[0::2] = (symbols.real > 0).astype(np.int8)
        out[1::2] = (symbols.imag > 0).astype(np.int8)
        return out
    raise ValueError("unknown scheme %r" % scheme)


def soft_llr(symbols, scheme, noise_var_per_dim):
    """Log likelihood ratios for the decoder, positive meaning a 1 is likelier.

    Scaling by the true noise variance matters. A Viterbi decoder with a
    correctly scaled soft input is worth roughly 2 dB over hard decisions; feed
    it unscaled correlator outputs and you keep some of that but not all, and
    the loss is invisible unless you measure against the hard decision curve.
    """
    if scheme == "bpsk":
        return 2.0 * symbols.real / noise_var_per_dim
    if scheme == "qpsk":
        out = np.empty(2 * len(symbols), dtype=float)
        scale = np.sqrt(2.0)
        out[0::2] = 2.0 * symbols.real * scale / noise_var_per_dim
        out[1::2] = 2.0 * symbols.imag * scale / noise_var_per_dim
        return out
    raise ValueError("unknown scheme %r" % scheme)


def awgn(symbols, ebn0_db, bits_per_symbol, rng, code_rate=1.0):
    """Add complex Gaussian noise for a given Eb/N0.

    Es/N0 = Eb/N0 * bits_per_symbol * code_rate. Symbols carry unit average
    energy, so N0 = Es / (Es/N0) and the per dimension variance is N0/2. The
    code rate term is the part that is easy to drop, and dropping it makes a
    rate 1/2 code look about 3 dB better than it is, because you have quietly
    stopped paying for the redundancy.
    """
    esn0 = 10.0 ** (ebn0_db / 10.0) * bits_per_symbol * code_rate
    n0 = 1.0 / esn0
    sigma = np.sqrt(n0 / 2.0)
    noise = rng.normal(0.0, sigma, len(symbols)) + 1j * rng.normal(
        0.0, sigma, len(symbols)
    )
    return symbols + noise, n0 / 2.0


def upsample(symbols, sps):
    out = np.zeros(len(symbols) * sps, dtype=complex)
    out[::sps] = symbols
    return out


def shape_and_match(symbols, sps, span, beta):
    """Full transmit shaping and receive matched filtering, sampled back at the
    symbol rate. Used to confirm the shaped chain agrees with the ideal
    symbol by symbol chain rather than assuming it does."""
    taps = rrc_taps(sps, span, beta)
    tx = np.convolve(upsample(symbols, sps), taps)
    rx = np.convolve(tx, taps)
    delay = len(taps) - 1
    return rx[delay : delay + len(symbols) * sps : sps]
