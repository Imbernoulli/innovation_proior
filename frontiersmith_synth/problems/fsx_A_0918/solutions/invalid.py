# TIER: invalid
"""
A plausible bug: pick each task's single highest-value desk independently and forget to
check remaining capacity at all. On instances with any real contention this piles multiple
tasks onto the same tightly-capacitated desk and blows its budget -> invalid answer -> 0.
"""
import sys, json


def main():
    inst = json.load(sys.stdin)
    N, M = inst["n_tasks"], inst["n_agents"]
    value = inst["value"]
    assign = [max(range(M), key=lambda k: value[i][k]) for i in range(N)]
    print(json.dumps({"assign": assign}))


if __name__ == "__main__":
    main()
