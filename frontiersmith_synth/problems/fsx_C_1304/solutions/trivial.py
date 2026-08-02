# TIER: trivial
"""Trivial candidate: pump the same small flat dose every day, regardless of
rain, tariff, stage, or the reservoir's actual state. "Just water it a
little each day" -- ignores every mechanism in the problem."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    maxirr = inst["params"]["max_irrig_per_day"]
    dose = min(maxirr, 4.5)
    print(json.dumps({"irrig": [dose] * T}))


if __name__ == "__main__":
    main()
