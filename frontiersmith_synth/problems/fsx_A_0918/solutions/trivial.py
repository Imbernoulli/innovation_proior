# TIER: trivial
"""
Do the safest possible thing: send every task to the last desk (index n_agents-1). Since the
instance always keeps that desk affordable for everyone, this is always feasible, but it
ignores value entirely.
"""
import sys, json


def main():
    inst = json.load(sys.stdin)
    N, M = inst["n_tasks"], inst["n_agents"]
    assign = [M - 1] * N
    print(json.dumps({"assign": assign}))


if __name__ == "__main__":
    main()
