# TIER: invalid
"""Broken policy: emits reserve arrays of the WRONG length (T-1 instead of T),
so the evaluator's shape check rejects the answer on every instance -> scores 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    ans = {
        "reserve_important_kwh": [0.0] * (T - 1),
        "reserve_low_kwh": [0.0] * (T - 1),
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
