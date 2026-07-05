# TIER: trivial
"""Flat schedule m_t = 1 for every epoch -- i.e. "I designed nothing, keep the
default (hot) base learning rate constant."  This reproduces the evaluator's own
internal flat-schedule baseline exactly, so every rooftop plot normalizes to ~0.1.
No warm-up, no annealing: on the noisy near-linear plots the hot LR keeps over-
shooting and on the hard nonlinear plots there is nothing left to help, so this is
the weak reference the score is measured against."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    E = int(inst["n_epochs"])
    print(json.dumps([1.0] * E))


main()
