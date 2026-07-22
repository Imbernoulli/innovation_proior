#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

A worn clocktower gear train couples an input winding-shaft to an output
arbor through a hidden mechanism: (a) a nominal ratio r = p/q realised by two
integer tooth counts (SMALL p, q), and (b) a backlash "play" of hidden
half-width D between the mating teeth -- the driven gear's contact point only
moves once the input has travelled far enough to take up the slack, and after
every direction reversal the slack must be taken up again from the OPPOSITE
edge (the classic hysteretic backlash / dead-zone operator). On top of that,
the escapement can only rest on a whole tooth, so the logged angle is rounded
to a small quantum tied to the tooth pitch (gear quantization) before sensor
noise is added. Each testId fixes a DIFFERENT hidden train (p, q, D).

The solver only ever SEES this TRAIN trace, recorded while the clocktower
smith slowly calibrates the mechanism by hand: a LONG, mostly one-directional
turn of the winding key with at most one or two brief direction reversals
(the smith checking his work and turning back). In that quasi-static regime
the backlash lag is taken up almost immediately and then stays essentially
constant, so the logged trace looks close to a single straight line. You will
be graded on the escapement's real behaviour: a FAST drive that reverses
direction many times, fully exposing the hidden state -- a naive curve fit to
the logbook extrapolates badly there.

STDOUT prints ONLY a header "<n_train> <test_id>" then n_train rows
"<input_angle> <output_angle>". The hidden p, q, D, quantum and RNG seed are
NEVER printed -- data rows only.
"""
import sys, random, math

RATIO_CHOICES = [
    (2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2), (3, 5), (5, 3),
    (4, 5), (5, 4), (5, 6), (6, 5), (5, 7), (7, 5), (2, 7), (7, 2),
    (3, 7), (7, 3), (4, 7), (7, 4), (5, 8), (8, 5), (3, 8), (8, 3),
]
AMP_LO, AMP_HI = 0.70, 1.00
AMP_REF = 0.85       # fixed reference swing used to scale noise/quantum (NOT the random per-test amp)
SIGMA_FRAC = 0.18    # sensor-noise std, as a fraction of r*AMP_REF
QSTEP_FRAC = 0.08    # escapement quantum, as a fraction of r*AMP_REF


def hidden_params(t):
    """Hidden gear train for this test id (lives in gen AND grader, never printed)."""
    rng = random.Random(900037 + t * 7919)
    p, q = rng.choice(RATIO_CHOICES)
    D = rng.uniform(0.20, 0.45)
    return p, q, D


def simulate(xs, r, D, sigma, qstep, seed):
    """Backlash play operator: contact c tracks x within a band of half-width D;
    raw = r*c is rounded to the escapement quantum, then sensor noise is added.
    c starts from a fallback contact of 0.0 (mirrors the solver-side default
    Sk1=0 at t=0)."""
    rng = random.Random(seed)
    c_prev = 0.0
    ys = []
    for x in xs:
        lo, hi = x - D, x + D
        c = hi if c_prev > hi else (lo if c_prev < lo else c_prev)
        raw = r * c
        if qstep > 1e-12:
            raw = round(raw / qstep) * qstep
        ys.append(raw + rng.gauss(0.0, sigma))
        c_prev = c
    return ys


def slow_drive(t, n):
    """SLOW, mostly one-directional calibration turn: one DOMINANT segment
    (70-85% of the samples) plus 0-2 SHORT excursion reversals (the smith
    checking his work), covering a wide swing of the shaft."""
    rng = random.Random(31337 + t * 104729)
    n_extra = rng.choice([0, 0, 1, 1, 2])
    amp = rng.uniform(AMP_LO, AMP_HI)
    start = rng.uniform(-amp, amp)
    main_end = rng.uniform(-amp, amp)
    tries = 0
    while abs(main_end - start) < 0.7 * amp and tries < 50:
        main_end = rng.uniform(-amp, amp)
        tries += 1
    pts = [start, main_end]
    for _ in range(n_extra):
        nxt = pts[-1] + rng.uniform(-0.35, 0.35) * amp
        nxt = max(-amp, min(amp, nxt))
        pts.append(nxt)
    n_seg = len(pts) - 1
    main_frac = rng.uniform(0.70, 0.85)
    if n_seg == 1:
        seg_fracs = [1.0]
    else:
        rest = 1.0 - main_frac
        each = rest / (n_seg - 1)
        seg_fracs = [main_frac] + [each] * (n_seg - 1)
    seg_lens = [max(4, int(round(f * n))) for f in seg_fracs]
    seg_lens[0] += n - sum(seg_lens)
    xs = []
    for s in range(n_seg):
        a, b = pts[s], pts[s + 1]
        L = seg_lens[s]
        for i in range(L):
            frac = (i / (L - 1)) if L > 1 else 1.0
            xs.append(a + (b - a) * frac)
    xs = xs[:n]
    while len(xs) < n:
        xs.append(xs[-1] if xs else 0.0)
    return xs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = 340 - 6 * (t - 1)
    p, q, D = hidden_params(t)
    r = p / q
    sigma = SIGMA_FRAC * r * AMP_REF
    qstep = QSTEP_FRAC * r * AMP_REF
    xs = slow_drive(t, n)
    ys = simulate(xs, r, D, sigma, qstep, 555 + t * 13)
    out = ["%d %d" % (n, t)]
    for i in range(n):
        out.append("%.6f %.6f" % (xs[i], ys[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
