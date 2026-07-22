# TIER: trivial
import sys, json


def main():
    inst = json.load(sys.stdin)
    L = inst["lead_time"]
    all_demand = []
    for tl in inst["timelines"]:
        all_demand.extend(tl["demand"])
    mu = sum(all_demand) / max(1, len(all_demand))
    base_target = mu * L * 1.2  # thin, no statistically-sized safety margin
    ans = {
        "base_target": base_target,
        "trigger": 1e9,       # never fires
        "hoard_target": base_target,
        "cooldown_days": 0,
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
