# TIER: invalid
"""Deliberately broken: non-finite kernel entries and a bias that blows the
L2 budget. Must be rejected -> score 0 on every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    Lmax = inst["L_max"]
    n = inst["N"]
    kernel = [float("nan")] * (Lmax + 1)
    bias = [1e9] * n
    print(json.dumps({"kernel": kernel, "bias": bias}))


main()
