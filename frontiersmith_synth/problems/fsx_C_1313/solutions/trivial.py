# TIER: trivial
"""Flat, non-adaptive compromise: ramp the whole way to target_temp in ONE segment at a
fixed 45% of the burner's max rate.  Doesn't look at bands, thickness, or fuel cost at all
-- just picks a single "cautious-ish" number for the entire firing.  Pays a full-schedule
slowdown for safety it only needed inside two narrow bands, so it wastes a lot of fuel."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    start = float(inst["start_temp"])
    target = float(inst["target_temp"])
    max_rate = float(inst["max_rate"])
    rate = 0.45 * max_rate
    minutes = (target - start) / rate
    print(json.dumps([{"to_temp": target, "minutes": minutes}]))


main()
