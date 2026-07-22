# TIER: trivial
"""Do nothing: identity kernel (no spatial coupling at all), zero bias.
This exactly reproduces the evaluator's own "naive" reference, so it anchors
the score at ~0.1 on every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    Lmax = inst["L_max"]
    n = inst["N"]
    kernel = [1.0] + [0.0] * Lmax
    bias = [0.0] * n
    print(json.dumps({"kernel": kernel, "bias": bias}))


main()
