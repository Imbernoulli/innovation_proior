# TIER: invalid
"""Broken schedule: only ever ramps to HALF of target_temp, so it never reaches
target_temp -- the evaluator must reject this on every instance (score 0)."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    start = float(inst["start_temp"])
    target = float(inst["target_temp"])
    max_rate = float(inst["max_rate"])
    half = start + (target - start) * 0.5
    minutes = (half - start) / max_rate
    print(json.dumps([{"to_temp": half, "minutes": minutes}]))


main()
