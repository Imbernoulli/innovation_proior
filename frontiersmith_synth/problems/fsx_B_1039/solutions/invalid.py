# TIER: invalid
"""Emits an out-of-range crop index on every plot/season -- must score 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    P, T = inst["P"], inst["T"]
    plan = [[999999] * T for _ in range(P)]
    print(json.dumps({"plan": plan}, separators=(" , ", ": ")))


if __name__ == "__main__":
    main()
