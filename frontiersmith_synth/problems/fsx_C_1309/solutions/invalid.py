# TIER: invalid
"""Malformed candidate: wrong shape (missing a ward row) and a non-numeric
entry, plus a negative value -- must be rejected outright, not clipped."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    K = len(inst["wards"])
    reserve = [[1.0] * T for _ in range(K - 1)]     # missing a ward row
    reserve.append([-5.0] + ["lots"] * (T - 1))       # negative + non-numeric
    print(json.dumps({"reserve": reserve}))


if __name__ == "__main__":
    main()
