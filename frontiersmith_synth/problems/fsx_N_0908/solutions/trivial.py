# TIER: trivial
"""Do nothing smart: split every round's review budget evenly across all categories,
ignoring both the observed volume and the migration graph."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    K, T, R = inst["K"], inst["T"], inst["R"]
    alloc = [[R[t] / K] * K for t in range(T)]
    print(json.dumps({"alloc": alloc}))


main()
