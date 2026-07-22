# TIER: trivial
# Feedforward-only: dispatch a constant reserve equal to the forecast's mean. No feedback at all,
# no awareness of dynamics, delay, or the instability threshold.
import sys
import json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    d = inst["d_forecast"]
    m = sum(d) / T if T else 0.0
    print(json.dumps({"u": [m] * T}))


if __name__ == "__main__":
    main()
