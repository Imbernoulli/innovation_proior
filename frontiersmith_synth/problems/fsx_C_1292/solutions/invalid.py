# TIER: invalid
"""Broken candidate: claims to spend a ward on the very first descent regardless of whether it
has one, which is semantically invalid whenever wards_start == 0... but every instance here
starts with >=1 ward, so instead this candidate goes further and repeatedly claims to drink
potions well past its starting stock (spamming far more drink_potion actions than
potions_start), which is always infeasible. The evaluator must reject the whole run -> 0 on
every instance."""
import sys
import json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    # Ask for a ward-shielded descent, then drink far more potions than we could possibly have.
    actions = [{"action": "descend", "use_ward": True}]
    actions += [{"action": "drink_potion"} for _ in range(max(1, T))]
    print(json.dumps(actions))


main()
