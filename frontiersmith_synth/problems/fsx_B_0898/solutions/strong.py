# TIER: strong
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
    timelines = inst["timelines"]

    all_demand = []
    all_signal = []
    for tl in timelines:
        all_demand.extend(tl["demand"])
        all_signal.extend(tl["precursor_signal"])

    mu_d, sd_d = mean_std(all_demand)
    # calm-timeline-efficient base target: same lean recipe as the obvious
    # textbook approach -- this is where the insurance premium gets funded.
    base_target = mu_d * (L + 1) + 1.2 * sd_d * math.sqrt(L + 1)

    # the insight: read the sweep's OWN leading-indicator distribution to set
    # a trigger that is statistically separated from background chatter,
    # instead of guessing a fixed number or reacting to raw demand.
    mu_s, sd_s = mean_std(all_signal)
    trigger = mu_s + 2.0 * sd_s

    # size the hoard buffer from the demand data to cover a plausible shock
    # duration (we don't know the true embargo length, so budget for a
    # generously long one on top of the base target).
    shock_days = 7
    hoard_target = base_target + mu_d * shock_days

    # the SECOND half of the insight: once the buffer is built, do NOT keep
    # re-ordering through an unknown-length blackout -- a lingering elevated
    # target just keeps placing orders that are either lost in the embargo or
    # land in a costly pile-up right after it lifts. A short cooldown (just
    # long enough to finish the pre-embargo build-up) lets the pre-built
    # stock coast through the shock on its own; order-up-to logic naturally
    # stops reordering once on-hand already exceeds the (reverted) base
    # target, so nothing is wasted chasing a blocked border.
    cooldown_days = 4

    ans = {
        "base_target": base_target,
        "trigger": trigger,
        "hoard_target": hoard_target,
        "cooldown_days": cooldown_days,
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
