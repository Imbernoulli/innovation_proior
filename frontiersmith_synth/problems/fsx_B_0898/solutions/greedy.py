# TIER: greedy
import sys, json, math


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(max(0.0, var))


def main():
    inst = json.load(sys.stdin)
    L = inst["lead_time"]
    all_demand = []
    for tl in inst["timelines"]:
        all_demand.extend(tl["demand"])
    mu, sd = mean_std(all_demand)
    # textbook base-stock formula with a standard safety margin -- completely
    # blind to the precursor signal and to embargoes.
    base_target = mu * (L + 1) + 1.2 * sd * math.sqrt(L + 1)
    ans = {
        "base_target": base_target,
        "trigger": 1e9,     # never hoards
        "hoard_target": base_target,
        "cooldown_days": 0,
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
