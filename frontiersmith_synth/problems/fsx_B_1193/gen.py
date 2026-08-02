#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

A shape-memory alloy strip is loaded (stretched) and unloaded (released) along
one path.  Its restoring force y depends on the current elongation x AND on
which BRANCH of the loop the strip is currently traversing -- loading
(x increasing since the previous sample) or unloading (x decreasing).  Each
testId fixes a DIFFERENT hidden strip (its own centerline curve and branch
gap).  The branch flips exactly when the elongation's direction reverses; this
switching rule is FIXED and stated in the problem -- it is not part of what
must be discovered.

The solver only ever SEES this TRAIN trace: elongation/force pairs recorded
along a path with FEW reversals (mostly one simple up-down loop) and a
moderate elongation range.  The held-out grading path (regenerated only
inside the grader) revisits the SAME strip along a DIFFERENT, more agitated
path -- more reversals, a different sampling rate, and a wider elongation
range -- so any model that ignores which branch it is on, or that secretly
leans on sample position/spacing instead of the branch itself, is exposed.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"<x> <y>".  The hidden coefficients and the seed are NOT printed.
"""
import sys, random


def hidden_params(t):
    """Hidden strip law for this test id (duplicated in gen.py AND verify.py,
    never imported -- the ground truth lives only inside each script)."""
    rng = random.Random(900001 + 7919 * t)
    k0 = rng.uniform(-0.25, 0.25)
    k1 = rng.uniform(0.60, 1.30)
    k2 = rng.uniform(-0.35, 0.35)
    k3 = rng.uniform(-0.45, 0.45)
    c0 = rng.uniform(0.18, 0.32)
    c1 = rng.uniform(0.05, 0.20)
    sigma = 0.010 + 0.0012 * (t - 1)
    return k0, k1, k2, k3, c0, c1, sigma


def centerline(x, k0, k1, k2, k3):
    return k0 + k1 * x + k2 * x * x + k3 * x * x * x


def gap(x, c0, c1):
    return c0 + c1 * x * x


def branch_states(xs):
    """Fixed, stated switching rule: b[0]=+1 (loading); thereafter b flips to
    +1/-1 with the sign of x[i]-x[i-1], and HOLDS its previous value on an
    exact tie (never triggered by our own paths, kept for well-definedness)."""
    b = [1]
    for i in range(1, len(xs)):
        if xs[i] > xs[i - 1]:
            b.append(1)
        elif xs[i] < xs[i - 1]:
            b.append(-1)
        else:
            b.append(b[-1])
    return b


def true_series(xs, params, noise_seed):
    k0, k1, k2, k3, c0, c1, sigma = params
    b = branch_states(xs)
    rng = random.Random(noise_seed)
    ys = []
    for x, bi in zip(xs, b):
        y = centerline(x, k0, k1, k2, k3) + bi * gap(x, c0, c1) + rng.gauss(0.0, sigma)
        ys.append(y)
    return ys


def make_path(rng, n_segments, pts_per_seg, x_lo, x_hi, x_start):
    """Build ONE traversal path: n_segments monotonic runs alternating
    direction, turning points placed inside [x_lo,x_hi], each run sampled
    with slightly irregular (non-uniform) spacing -- the sampling RATE is
    incidental, only the sequence of directions/turning points matters."""
    xs = [x_start]
    cur = x_start
    going_up = True
    span = x_hi - x_lo
    for _ in range(n_segments):
        # turning points sit near the domain EXTREMES (with jitter) so the
        # traversed path actually spans the amplitude, like a real loop --
        # a purely uniform-random target can cluster and starve the fit of
        # x-range, making any cubic fit (greedy or strong) ill-conditioned.
        if going_up:
            target = rng.uniform(x_hi - 0.20 * span, x_hi)
        else:
            target = rng.uniform(x_lo, x_lo + 0.20 * span)
        # keep runs from being degenerately short
        if going_up and target <= cur:
            target = cur + rng.uniform(0.15, 0.4)
        if (not going_up) and target >= cur:
            target = cur - rng.uniform(0.15, 0.4)
        target = min(x_hi, max(x_lo, target))
        m = pts_per_seg()
        # irregular fractional offsets along the run (non-uniform rate)
        fracs = sorted(rng.uniform(0.0, 1.0) for _ in range(m))
        fracs = [(0.06 + 0.88 * f) for f in fracs]  # keep off exact endpoints
        for f in fracs:
            v = cur + (target - cur) * f
            if not xs or abs(v - xs[-1]) > 1e-4:
                xs.append(round(v, 6))
        xs.append(round(target, 6))
        cur = target
        going_up = not going_up
    # de-duplicate accidental exact repeats from rounding, preserving order
    out = [xs[0]]
    for v in xs[1:]:
        if abs(v - out[-1]) > 1e-4:
            out.append(v)
    return out


def train_path(t):
    """Few-reversal, moderate-range TRAIN path; density ladders with t."""
    rng = random.Random(31 + t * 104729)
    if t <= 3:
        n_seg, lo, hi = 1, -0.55, 0.55
    elif t <= 6:
        n_seg, lo, hi = 2, -0.70, 0.70
    else:
        n_seg, lo, hi = 3, -0.85, 0.85
    base = 16 + 3 * t
    xs = make_path(rng, n_seg, lambda: base + rng.randint(-2, 3), lo, hi, x_start=lo * 0.5)
    return xs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    xs = train_path(t)
    params = hidden_params(t)
    ys = true_series(xs, params, noise_seed=555 + t * 13)
    n = len(xs)
    out = ["%d %d" % (n, t)]
    for x, y in zip(xs, ys):
        out.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
