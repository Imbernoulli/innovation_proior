# TIER: trivial
"""Monoculture of crop 0 on every plot, every season -- ignores soil state,
pest memory and market glut entirely. Reproduces the checker's own baseline."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    P, T = inst["P"], inst["T"]
    plan = [[0] * T for _ in range(P)]
    print(json.dumps({"plan": plan}, separators=(" , ", ": ")))


if __name__ == "__main__":
    main()
