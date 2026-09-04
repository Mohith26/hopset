"""Frequency hopping: hopset derivation, lockouts, and jamming.

Band parameters follow the publicly documented VHF combat net radio case:
30.000 to 87.975 MHz in 25 kHz steps, which is 2320 channels, hopping in the
region of 100 hops per second.

Nothing classified is reimplemented here and nothing could be. The transmission
security property being modelled is structural, not cryptographic: two radios
holding the same key, the same net identifier, and the same time of day derive
the same channel sequence without ever sending the sequence over the air. I use
HMAC-SHA256 as the generator because it is public, it is a good pseudorandom
function, and the point of the exercise is what the sequence has to be like,
not how a particular military generator produces it.
"""

import hashlib
import hmac
import struct

import numpy as np

BAND_START_HZ = 30_000_000
CHANNEL_SPACING_HZ = 25_000
NUM_CHANNELS = 2320
HOP_RATE_HZ = 111.0
DWELL_S = 1.0 / HOP_RATE_HZ
HOPPING_BANDWIDTH_HZ = NUM_CHANNELS * CHANNEL_SPACING_HZ


def channel_hz(index):
    return BAND_START_HZ + index * CHANNEL_SPACING_HZ


def processing_gain_db():
    """10 log10 (spread bandwidth / instantaneous bandwidth).

    The anti jam advantage of hopping, in the same sense as a direct sequence
    system's spreading gain: a jammer that wants to sit on you everywhere has
    to divide its power across the whole hopping band.
    """
    return 10.0 * np.log10(HOPPING_BANDWIDTH_HZ / float(CHANNEL_SPACING_HZ))


class Hopset:
    """A derived channel sequence, with lockouts.

    Real deployments exclude channels for interference deconfliction and to
    protect frequencies in use by others. Lockouts are the reason the hopset
    cannot simply be an index into a fixed table: removing channels changes the
    modulus, and doing that naively biases the sequence toward low indices.
    Here the generator redraws instead, which keeps the distribution uniform
    over the allowed set at the cost of an occasional extra hash.
    """

    def __init__(self, key, net_id, lockout=None, num_channels=NUM_CHANNELS):
        self.key = key
        self.net_id = net_id
        self.num_channels = num_channels
        self.lockout = frozenset(lockout or ())
        self.allowed = num_channels - len(self.lockout)
        if self.allowed <= 0:
            raise ValueError("every channel is locked out")

    def _draw(self, tod, attempt):
        msg = struct.pack(">IQI", self.net_id, tod, attempt)
        digest = hmac.new(self.key, msg, hashlib.sha256).digest()
        return int.from_bytes(digest[:4], "big") % self.num_channels

    def channel_at(self, tod):
        for attempt in range(64):
            ch = self._draw(tod, attempt)
            if ch not in self.lockout:
                return ch
        raise RuntimeError("could not draw an allowed channel")

    def sequence(self, tod_start, length):
        return np.array(
            [self.channel_at(tod_start + i) for i in range(length)], dtype=np.int32
        )


def uniformity_chi_square(sequence, num_channels=NUM_CHANNELS, lockout=()):
    """Chi square statistic of the channel histogram against uniform.

    A hopset that revisits some channels more than others is a hopset a jammer
    can learn. This is a cheap sanity statistic, not a cryptographic claim.
    """
    allowed = [c for c in range(num_channels) if c not in set(lockout)]
    counts = np.bincount(sequence, minlength=num_channels)[allowed]
    expected = len(sequence) / float(len(allowed))
    return float(np.sum((counts - expected) ** 2) / expected), len(allowed) - 1


def jammed_set(kind, fraction, rng, num_channels=NUM_CHANNELS):
    """Which channels a jammer is sitting on this instant.

    partial_band  one contiguous block, the cheapest thing to build
    scattered     independent random channels, a smarter allocation of the
                  same total power
    """
    n = int(round(num_channels * fraction))
    if n <= 0:
        return set()
    if kind == "partial_band":
        start = rng.integers(0, num_channels)
        return set(int((start + i) % num_channels) for i in range(n))
    if kind == "scattered":
        return set(int(c) for c in rng.choice(num_channels, size=n, replace=False))
    raise ValueError("unknown jammer %r" % kind)


def follower_jammer_catches(reaction_s, dwell_s=DWELL_S):
    """A follower jammer listens, finds the current channel, retunes, and
    transmits. It only does damage for whatever is left of the dwell after its
    own reaction time, so the fraction of each hop it can corrupt is
    max(0, 1 - reaction/dwell). At 111 hops per second the dwell is 9.0 ms, so
    a jammer needing 10 ms to react accomplishes nothing at all."""
    return max(0.0, 1.0 - reaction_s / dwell_s)
