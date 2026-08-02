# TIER: greedy
"""Obvious first-instinct policy: maximize same-day bed utilization. Never hold
back any capacity -- if a bed is free, anyone waiting for it (home patient or
outlier) may take it. This is exactly the "admit whenever a bed is free" trap:
it never anticipates that a ward it just lent beds from will need them back
for its OWN patients soon, and outlier patients (mechanism 2) occupy the
borrowed bed for far longer than a home patient would."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    K = len(inst["wards"])
    reserve = [[0.0] * T for _ in range(K)]
    print(json.dumps({"reserve": reserve}))


if __name__ == "__main__":
    main()
