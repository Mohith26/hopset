"""Rate 1/2, constraint length 7 convolutional coding with a Viterbi decoder.

Generators are 171 and 133 octal, the pair used by essentially every tactical
and satellite link that reaches for a K=7 convolutional code, chosen because
they maximise the free distance (dfree = 10) at that constraint length.

The decoder is a soft decision Viterbi over all 64 states, vectorised across
states with numpy so a full add compare select stage is a handful of array
operations instead of a 64 iteration Python loop.

Two properties give this an oracle that does not depend on my own encoder:

  * At zero noise the decoder has to return the input bits exactly. That
    catches state numbering and traceback errors, which otherwise show up only
    as a mysteriously bad BER.
  * Soft decision decoding should beat hard decision decoding by roughly 2 dB
    on an AWGN channel. That number is a textbook result, not something I can
    tune, so if my soft path is not about 2 dB better than my hard path, the
    LLR scaling is wrong.
"""

import numpy as np

K = 7
N_STATES = 1 << (K - 1)
G = (0o171, 0o133)
RATE = 0.5


def _parity(x):
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def _build_trellis():
    """next_state, outputs, and the reverse edges the decoder needs.

    State holds the previous 6 input bits with the most recent in the high bit.
    On input b the 7 bit register is v = (b << 6) | state, the two output bits
    are the parities of v masked by each generator, and the new state is
    v >> 1.
    """
    next_state = np.zeros((N_STATES, 2), dtype=np.int32)
    out_bits = np.zeros((N_STATES, 2, 2), dtype=np.int8)
    for s in range(N_STATES):
        for b in (0, 1):
            v = (b << (K - 1)) | s
            next_state[s, b] = v >> 1
            for gi, g in enumerate(G):
                out_bits[s, b, gi] = _parity(v & g)
    # reverse edges: for each new state, its two predecessors and their inputs
    pred_state = np.zeros((N_STATES, 2), dtype=np.int32)
    pred_input = np.zeros((N_STATES, 2), dtype=np.int8)
    pred_out = np.zeros((N_STATES, 2, 2), dtype=np.int8)
    counts = np.zeros(N_STATES, dtype=np.int32)
    for s in range(N_STATES):
        for b in (0, 1):
            ns = next_state[s, b]
            slot = counts[ns]
            pred_state[ns, slot] = s
            pred_input[ns, slot] = b
            pred_out[ns, slot] = out_bits[s, b]
            counts[ns] += 1
    assert np.all(counts == 2), "every state must have exactly two predecessors"
    return next_state, out_bits, pred_state, pred_input, pred_out


NEXT_STATE, OUT_BITS, PRED_STATE, PRED_INPUT, PRED_OUT = _build_trellis()
# +1 for a transmitted 1, -1 for a transmitted 0, so a branch metric is a dot
# product with the received log likelihood ratios.
PRED_SIGN = (2.0 * PRED_OUT.astype(float) - 1.0)


def encode(bits, terminate=True):
    """Encode, optionally flushing 6 zero tail bits so the trellis ends in
    state 0. Termination costs a little rate and buys a decoder that knows
    where it finishes, which is worth more on short frames than the rate."""
    bits = np.asarray(bits, dtype=np.int8)
    if terminate:
        bits = np.concatenate([bits, np.zeros(K - 1, dtype=np.int8)])
    state = 0
    out = np.empty(2 * len(bits), dtype=np.int8)
    for i, b in enumerate(bits):
        out[2 * i] = OUT_BITS[state, b, 0]
        out[2 * i + 1] = OUT_BITS[state, b, 1]
        state = NEXT_STATE[state, b]
    return out


def decode(llrs, n_info_bits, terminated=True):
    """Soft decision Viterbi.

    `llrs` are per coded bit, positive meaning a 1 is more likely. For hard
    decision decoding pass +1/-1.
    """
    llrs = np.asarray(llrs, dtype=float)
    n_steps = len(llrs) // 2
    pair = llrs.reshape(n_steps, 2)

    NEG = -1e9
    pm = np.full(N_STATES, NEG)
    pm[0] = 0.0
    choices = np.empty((n_steps, N_STATES), dtype=np.int8)

    for t in range(n_steps):
        l0, l1 = pair[t, 0], pair[t, 1]
        branch = PRED_SIGN[:, :, 0] * l0 + PRED_SIGN[:, :, 1] * l1
        cand = pm[PRED_STATE] + branch
        best = np.argmax(cand, axis=1)
        choices[t] = best
        pm = cand[np.arange(N_STATES), best]

    state = 0 if terminated else int(np.argmax(pm))
    decoded = np.empty(n_steps, dtype=np.int8)
    for t in range(n_steps - 1, -1, -1):
        slot = choices[t, state]
        decoded[t] = PRED_INPUT[state, slot]
        state = PRED_STATE[state, slot]
    return decoded[:n_info_bits]


class BlockInterleaver:
    """Write by rows, read by columns.

    A convolutional code corrects scattered errors well and consecutive errors
    badly, and a frequency hopping radio produces exactly the wrong kind: when
    a hop lands on a jammed channel, every bit sent during that dwell is
    destroyed together. Interleaving across hops turns one long burst into many
    isolated errors spread over the codeword, which is the shape the decoder
    was designed for. This is the single largest effect measured anywhere in
    this project.
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols

    def interleave(self, bits):
        bits = np.asarray(bits)
        if len(bits) != self.size:
            raise ValueError("expected %d bits, got %d" % (self.size, len(bits)))
        return bits.reshape(self.rows, self.cols).T.reshape(-1).copy()

    def deinterleave(self, bits):
        bits = np.asarray(bits)
        if len(bits) != self.size:
            raise ValueError("expected %d bits, got %d" % (self.size, len(bits)))
        return bits.reshape(self.cols, self.rows).T.reshape(-1).copy()
