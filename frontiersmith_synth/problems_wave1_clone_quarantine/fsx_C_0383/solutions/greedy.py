# TIER: greedy
"""The obvious first fix: the base LR is hot, so just turn it DOWN with a flat
lower multiplier (m_t = 0.5 everywhere).  Halving the effective LR stops the worst
over-shoot, so this clearly beats the flat-1 baseline on several plots.  But a
constant lower LR is still the wrong SHAPE: it is now too slow to fully converge the
hard spiral plots within the epoch budget, and it still lacks late annealing to
settle the noisy plots.  No single constant multiplier wins every plot, so the
geometric mean leaves this well short of a properly shaped schedule."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    E = int(inst["n_epochs"])
    print(json.dumps([0.5] * E))


main()
