# TIER: trivial
"""Identity activation g(x)=x -- no nonlinearity at all.  This reproduces the
evaluator's own internal (linear) baseline, so every station normalizes to ~0.1.
A 2-layer MLP with a linear hidden activation collapses to a linear classifier,
which is exactly the reference the score is measured against."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    grid = inst["grid"]
    g = [float(x) for x in grid]          # g(x) = x  (sampled at the grid knots)
    print(json.dumps(g))


main()
