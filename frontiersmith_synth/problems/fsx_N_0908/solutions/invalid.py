# TIER: invalid
"""Deliberately infeasible: blows the per-round budget and emits a negative allocation.
Must be rejected by the evaluator and scored 0.0 on every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    K, T = inst["K"], inst["T"]
    alloc = [[-1.0] + [1e9] * (K - 1) for _ in range(T)]
    print(json.dumps({"alloc": alloc}))


main()
