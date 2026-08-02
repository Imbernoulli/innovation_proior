# TIER: trivial
"""Reckless descent: always try to go one floor deeper, every turn, never grinding, never
drinking a potion, never spending a ward. This is EXACTLY the evaluator's own internal
reference construction (see evaluator.py:baseline), so it normalizes to ~0.1 on every
instance by construction: it embodies half of the trap named in the brief ("descending as
fast as possible... eventually meets an unsurvivable floor") -- it dies on nearly every
instance and banks nothing."""
import sys
import json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    actions = [{"action": "descend", "use_ward": False} for _ in range(T)]
    print(json.dumps(actions))


main()
