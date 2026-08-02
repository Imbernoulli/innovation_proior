# TIER: invalid
"""Invalid candidate: emits a schedule that blows past the daily pump cap on
purpose. Must be rejected (score 0) by the evaluator's strict validation."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    maxirr = inst["params"]["max_irrig_per_day"]
    # deliberately out of range -- more than the daily cap every day
    print(json.dumps({"irrig": [maxirr * 5.0 + 100.0] * T}))


if __name__ == "__main__":
    main()
