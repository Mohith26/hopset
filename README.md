# Hopset

A frequency hopping radio waveform and the real time scheduling it has to run
inside, both simulated end to end and both checked against closed form theory
rather than against themselves.

Band parameters follow the publicly documented VHF combat net case: 30.000 to
87.975 MHz in 25 kHz steps, 2320 channels, hopping at 111 hops per second, so a
dwell is 9.0 milliseconds. Nothing classified is reimplemented here and nothing
could be. The transmission security property being modelled is structural: two
radios holding the same key, net identifier, and time of day derive the same
channel sequence without ever sending the sequence over the air. I use
HMAC-SHA256 as the generator because it is public and it is a good
pseudorandom function.

Two things came out of building this that I did not expect, and both are the
reason the repo is worth reading:

1. **The naive receiver makes frequency hopping worse than not hopping**, at
   every level of jamming I tested. Not marginally. At 10% of the band jammed
   the hopping link failed 58% of frames and a single fixed channel failed 8.7%
   of them.
2. **A textbook blocking bound I wrote from memory was wrong**, and the
   scheduler simulator caught it by disagreeing on a task that has no critical
   section at all.

## Layout

```
hopset/phy.py        pulse shaping, AWGN, matched filter, Q function
hopset/fec.py        rate 1/2 K=7 convolutional code, soft Viterbi, interleaver
hopset/hop.py        hopset derivation, lockouts, jammer models
hopset/timing.py     clock drift and serial search acquisition
hopset/rt.py         fixed priority scheduling, response time analysis, PIP
hopset/waveform.py   the whole chain, four receiver models
hopset/experiments.py
hopset/test_hopset.py
```

numpy only. `python -m hopset.test_hopset`, `python -m hopset.experiments <section>`.

## The baseband is calibrated, not just self consistent

A DSP chain that is subtly wrong still runs, still produces a curve, and the
curve still bends the right way. It is just in the wrong place, and nothing
tells you. So the bit error rate is checked against `Q(sqrt(2 Eb/N0))`, the
closed form for coherent BPSK over AWGN, at 2,000,000 bits per point:

| Eb/N0 | measured BPSK | theory | measured QPSK |
| --- | --- | --- | --- |
| 0 dB | 0.078643 | 0.078650 | 0.078900 |
| 2 dB | 0.037551 | 0.037506 | 0.037412 |
| 4 dB | 0.012430 | 0.012501 | 0.012463 |
| 6 dB | 0.0023655 | 0.0023883 | 0.002474 |
| 8 dB | 0.00018350 | 0.00019091 | 0.00017950 |

Worst relative deviation across all ten points: **6.0%**, at the 8 dB point
where only about 380 errors occur in two million bits.

QPSK landing on the same curve is its own check. Gray mapped QPSK is two
independent BPSK channels in quadrature, so it carries twice the bits in the
same bandwidth at identical energy per bit, and any error in the quadrature
handling shows up as a curve 3 dB off rather than on top.

The transmit and receive filters are root raised cosines, so the cascade is a
full raised cosine and satisfies Nyquist. Peak intersymbol interference of the
cascade measures below 5e-3 for beta from 0.2 to 0.5, and the residual after a
full shape and match round trip truncated to 12 symbols is 5.9e-3.

## What the code actually buys

Rate 1/2, constraint length 7, generators 171 and 133 octal, soft decision
Viterbi over all 64 states, vectorised across states with numpy. Measured at a
bit error rate of 1e-3, with the noise scaled by the code rate in every case so
the x axis is energy per information bit and the code pays for its own
redundancy:

```
uncoded BPSK          6.77 dB
hard decision Viterbi 4.55 dB     coding gain 2.22 dB
soft decision Viterbi 2.92 dB     coding gain 3.85 dB
```

Soft over hard comes out at **1.63 dB**. The number usually quoted is about
2 dB, and that is an asymptotic figure at much lower error rates, so measuring
1.63 dB at 1e-3 is the expected shape rather than a discrepancy. It is also the
sharpest available check that the log likelihood ratios are scaled by the true
noise variance and not by something proportional to it: get the scaling wrong
and soft decoding keeps some of its advantage but not all of it, and nothing
else in the system complains.

## The result that stopped me: hopping made things worse

The end to end frame is 506 information bits, encoded to 1024 coded bits,
interleaved across 32 hops, 288 milliseconds on the air. Jamming is partial
band noise: a hop landing on a jammed channel is received at -6 dB instead of
6 dB.

My first receiver computed log likelihood ratios with the nominal noise
variance, which is what you get if nobody has thought about jamming. Frame
error rate, 150 frames per point:

| band jammed | fixed channel | hopping, nominal receiver | hopping, no interleaving |
| --- | --- | --- | --- |
| 2% | 0.007 | 0.060 | 0.453 |
| 5% | 0.027 | 0.260 | 0.767 |
| 10% | 0.087 | 0.580 | 0.967 |
| 20% | 0.167 | 0.960 | 1.000 |
| 35% | 0.260 | 1.000 | 1.000 |

A fixed channel beat the hopping link at **every** jammed fraction. That is the
opposite of the entire reason the waveform exists, so either the simulation was
broken or I had missed something.

The simulation was fine. The mechanism is this. A fixed channel radio is a coin
flip: it is either clean, and everything gets through, or jammed, and nothing
does. A hopping radio spreads one frame over 32 channels, so at 10% jamming
roughly 3.4 hops per frame are hit and about a tenth of the coded bits are
destroyed. That trade is supposed to be good, because a rate 1/2 code with
interleaving eats scattered errors for breakfast. It is only good if the
decoder is told those bits are unreliable.

A jammed sample is not a small number. It is a large number with the wrong
sign. Scaled by the nominal noise variance it becomes a large magnitude log
likelihood ratio, which the Viterbi decoder reads as high confidence. Ten
percent of the codeword arriving as confident lies is far more damaging than
ten percent arriving as erasures, and it is enough to drag the survivor path
away from the truth across the whole frame. Interleaving, which is supposed to
help, actively spreads the poison evenly through the codeword instead of
confining it to one burst the decoder might have ridden out.

The fix does not need to know anything. Under BPSK all the signal energy sits
on the in phase arm, so the quadrature arm of each hop is a free, clean noise
sample. Estimating the noise variance separately on each hop from its own
quadrature samples costs 32 multiplies per hop and requires no side
information, no training sequence, and no jam state detector:

| band jammed | fixed channel | nominal | LLR clipped | per hop estimate | oracle erasure |
| --- | --- | --- | --- | --- | --- |
| 2% | 0.007 | 0.060 | 0.000 | **0.000** | 0.000 |
| 5% | 0.027 | 0.260 | 0.013 | **0.000** | 0.000 |
| 10% | 0.087 | 0.580 | 0.067 | **0.000** | 0.000 |
| 20% | 0.167 | 0.960 | 0.407 | **0.000** | 0.020 |
| 35% | 0.260 | 1.000 | 0.873 | **0.153** | 0.293 |
| 50% | 0.453 | 1.000 | 1.000 | **0.720** | 0.833 |

The blind estimator turns a waveform that lost to a fixed channel everywhere
into one that wins everywhere up to half the band being jammed.

The other thing in that table is worth pausing on. From 20% jamming upward the
blind estimator beats the **oracle**, the receiver that is simply told which
hops were jammed and erases them. Erasure throws a hop away completely. The
estimator keeps it at correctly reduced confidence, and a jammed hop at -6 dB
still carries a little information. Being told the answer is not the same as
being told the right question, and hard erasure is the wrong question.

Also measured: processing gain over the 58 MHz hopping band is **33.65 dB**,
and a follower jammer has to react inside the 9.0 ms dwell to accomplish
anything at all. A jammer needing 1 ms corrupts 88.9% of a dwell, one needing
4.5 ms corrupts 50.1%, and one needing 10 ms corrupts nothing.

## Real time: the analysis I got wrong

A hopping radio is a hard real time system before it is anything else. The
synthesiser must be retuned and settled inside every 9 ms dwell, and missing
that is not graceful degradation, it is being off the net.

`rt.py` has a fixed priority preemptive simulator and a separate response time
analysis, written independently so they can disagree. They are supposed to
disagree in exactly one direction: no observed response time may ever exceed
the analytic bound.

They disagreed in the other direction. On a five task radio set, the
`keystream` task showed a worst observed response of 27 ticks against an
analytic bound of 10. A simulator beating a sound bound means the bound is
wrong, and it was.

My blocking term only counted **direct** blocking: a task is delayed by a
shared resource only if it uses that resource itself. `keystream` has no
critical section at all, so I gave it a blocking term of zero. But when the low
priority built in test task holds the synthesiser control resource and the high
priority hop timer waits on it, priority inheritance raises the holder to the
hop timer's priority. It now outranks `keystream`, which has nothing to do with
the resource and gets pushed aside by a critical section it has no relationship
to. That is **push-through blocking**, and the correct condition is about the
resource rather than the task: a lower priority critical section can block task
i if the resource it guards is used by any task at least as high priority as i.

With that corrected, `keystream` gets a 24 tick blocking term and a bound of 34
against 27 observed, and every task in the set sits under its bound.

The radio task set, in ticks of 100 microseconds, utilisation 0.681 against a
Liu and Layland bound of 0.744 for five tasks:

| task | period | WCET | blocking | analytic R | observed R | deadline |
| --- | --- | --- | --- | --- | --- | --- |
| hop_timer | 90 | 4 | 24 | 28 | 21 | 90 |
| keystream | 90 | 6 | 24 | 34 | 27 | 90 |
| modem_frame | 200 | 100 | 24 | 144 | 120 | 200 |
| operator_ui | 500 | 20 | 24 | 164 | 140 | 500 |
| built_in_test | 1000 | 30 | 0 | 170 | 170 | 1000 |

The hop timer needs 4 ticks of work and has a worst case response of 28,
because 24 of those are spent waiting on a built in test routine that holds the
synthesiser. Six times its own execution time, spent waiting on the least
important task in the system. That is what a blocking term looks like when you
write it down.

To check the analysis against something outside my own code, `experiments.py
scheduling` also runs a standard worked example: three tasks at utilisation
0.929, well above the Liu and Layland bound of 0.780, so the utilisation test
rejects it. Response time analysis accepts it, giving 3, 6, and 20 ticks, with
the lowest priority task landing exactly on its 20 tick deadline. The simulator
reproduces all three response times **exactly**, zero pessimism, which is the
expected result when every task is released together at a critical instant.

## Priority inversion

The Mars Pathfinder shape, reproduced on demand. A high priority task blocked
by a low priority task's 15 tick critical section, with an unrelated 30 tick
medium priority task in between that never touches the resource.

```
                        worst response   worst blocking
without inheritance         47 ticks         41 ticks
with inheritance            17 ticks         11 ticks
```

Without inheritance the medium task preempts the resource holder, so the high
priority task waits for the critical section *and* the entire medium priority
task. With inheritance the holder runs at the waiter's priority and the medium
task cannot cut in front, so the wait collapses to the critical section itself.

## Net timing

Two radios agree on the hop sequence only if they agree on time of day, so
drift sets the resynchronisation schedule. Drift is linear, which makes this
exact: seconds to slide outside the guard interval is guard / ppm.

| oscillator | 200 us guard | 500 us guard |
| --- | --- | --- |
| 0.5 ppm | 400 s | 1000 s |
| 2 ppm | 100 s | 250 s |
| 10 ppm | 20 s | 50 s |

A joiner that does not know the net's time of day has to search, which is a
serial search over uncertainty cells with a detection probability, a false
alarm probability, and a penalty for chasing false alarms. That has a standard
mean acquisition time, and the simulator reproduces it:

| clock error | cells | closed form | simulated | p95 |
| --- | --- | --- | --- | --- |
| 50 ms | 13 | 0.089 s | 0.089 s | 0.270 s |
| 270 ms | 61 | 0.406 s | 0.412 s | 1.037 s |
| 1 s | 223 | 1.477 s | 1.441 s | 3.595 s |
| 5 s | 1111 | 7.343 s | 7.247 s | 18.79 s |

Worst relative error 2.4%. The p95 column is the operationally interesting one:
at 5 seconds of clock uncertainty the average joiner is on the net in 7 seconds
and one in twenty is still searching after 19.

## Verification

`python -m hopset.test_hopset`: **25 cases, 211 assertions, 0 failures, 10.5s.**

Checked against something written by somebody else wherever possible: the Q
function against tabulated values to 1e-5, the bit error rate against the
closed form, the response time analysis against a published worked example, the
acquisition simulator against the standard serial search mean, and the coding
gain against the textbook soft over hard figure. The hopset lockout test uses a
chi square statistic to confirm that excluding channels by redrawing keeps the
distribution uniform, which a modulo remap would not.

## What I left out

- No carrier or timing recovery loop. Coherent detection is assumed, so this
  measures the waveform's own limits and not a synchroniser's.
- The channel is AWGN. Real VHF ground to ground is not, and a fading channel
  would change the interleaver depth conversation entirely.
- One shared resource in the scheduling model. Real blocking analysis with
  several nested resources needs the priority ceiling protocol, not plain
  inheritance.
- Hop timing is modelled as perfect once acquired. Settling time and
  transmit/receive turnaround inside the dwell are real and are not here.
