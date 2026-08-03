# TIER: strong
"""A SHAPED schedule: a brief linear warm-up followed by cosine annealing down to a
small floor.

Why it generalizes across every rooftop where any constant multiplier does not.
Early on it runs the LR near the (hot) base value, so the hard nonlinear spiral and
xor/rings plots make fast progress and actually converge within the fixed epoch
budget -- something a low flat LR cannot do.  It then cosine-anneals toward ~0.02,
which lets the noisy near-linear plots settle instead of forever over-shooting the
label noise the way the hot flat baseline does.  The short warm-up avoids the first-
step instability of hitting the hot LR cold.  Because it never collapses on any
single plot, the geometric mean rewards it far more than any flat schedule.  It
still leaves headroom -- no plot is solved perfectly, so a better-tuned peak / floor
/ anneal shape could score higher."""
import sys, json
import math


def main():
    inst = json.load(sys.stdin)
    E = int(inst["n_epochs"])
    warm = max(1, E // 15)          # short warm-up
    hi, lo = 1.0, 0.02
    sched = []
    for t in range(E):
        if t < warm:
            m = hi * (t + 1) / warm
        else:
            p = (t - warm) / max(1, (E - warm))
            m = lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * p))
        sched.append(float(m))
    print(json.dumps(sched))


main()
