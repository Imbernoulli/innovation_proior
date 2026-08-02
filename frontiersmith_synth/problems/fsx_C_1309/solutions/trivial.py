# TIER: trivial
"""Naive reference: never lend a bed across wards at all. Every ward's reserve is
set to its full capacity every day, so an outlier admission is never possible --
each patient can only be treated in their own home ward, or boards. No forecast
use, no cross-ward reasoning, no discharge-timing awareness."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    capacity = inst["capacity"]
    reserve = [[float(c)] * T for c in capacity]
    print(json.dumps({"reserve": reserve}))


if __name__ == "__main__":
    main()
