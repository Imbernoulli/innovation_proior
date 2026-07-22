# TIER: trivial
import sys, json


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]
    K = inst["K"]
    # One deterministic full pass so every gauge gets its mandatory single
    # visit, then waste the entire surplus budget re-updating gauge 0 (which
    # is already converged and never needs it) -- no thought given to who
    # actually needs the extra visits.
    order = list(range(n))
    if K > n:
        order += [0] * (K - n)
    order = order[:K]
    print(json.dumps({"order": order}))


main()
