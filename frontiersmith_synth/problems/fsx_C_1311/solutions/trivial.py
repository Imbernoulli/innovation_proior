# TIER: trivial
"""Always shed every non-critical tier: reserve floor = capacity_kwh on every day
for both 'important' and 'low'. Ultra-conservative -- critical is always fully
protected, but almost all important/low demand is forfeited even on days with
abundant sun, since the reserve floor is never relaxed. This reproduces the
evaluator's own internal weak baseline (obj_base), so it anchors to r ~ 0.1 on
every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    cap = float(inst["capacity_kwh"])
    ans = {
        "reserve_important_kwh": [cap] * T,
        "reserve_low_kwh": [cap] * T,
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
