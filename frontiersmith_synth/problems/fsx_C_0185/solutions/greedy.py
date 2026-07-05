# TIER: greedy
"""ReLU: g(x)=max(0,x).  The obvious textbook nonlinearity -- it clearly beats the
linear baseline on the sharply nonlinear stations (xor, rings, bands, spiral, and
the held-out xor), which is a big improvement.  But a hard rectifier is unbounded
and non-saturating, so on the wavy sinusoidal-boundary station (where the boundary
is nearly linear and the labels are noisy) it overfits and does WORSE than the
linear baseline -- that station collapses to the floor, and because the final
score is a geometric mean, ReLU lands well short of a saturating activation that
generalizes everywhere."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    grid = inst["grid"]
    g = [float(x) if x > 0.0 else 0.0 for x in grid]
    print(json.dumps(g))


main()
