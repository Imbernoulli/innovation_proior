# TIER: strong
import sys


def read_ints(tokens, k):
    return [int(next(tokens)) for _ in range(k)]


def tier_cost(supplier, q):
    if q <= 0:
        return 0
    price = supplier["tiers"][0][1]
    for th, pr in supplier["tiers"]:
        if q >= th:
            price = pr
        else:
            break
    return price * q


def main():
    toks = iter(sys.stdin.read().split())
    T, N, G = read_ints(toks, 3)
    V, P = read_ints(toks, 2)
    D = read_ints(toks, T)
    suppliers = []
    for _ in range(N):
        group, qualified, lead, qualcost, ntiers = read_ints(toks, 5)
        tiers = sorted(tuple(read_ints(toks, 2)) for _ in range(ntiers))
        suppliers.append(dict(group=group, qualified=qualified, lead=lead,
                               qualcost=qualcost, tiers=tiers))
    E = int(next(toks))
    disruptions = set()
    for _ in range(E):
        per, grp = read_ints(toks, 2)
        disruptions.add((per, grp))

    # The insight: single-sourcing the cheapest supplier is genuinely right most
    # of the time -- concentrating an order is what unlocks the deep volume-
    # discount tiers. The exposure is entirely the correlated-failure risk on the
    # periods that supplier's own group gets disrupted. Reformulate the decision:
    # is there a *different-group* supplier that, qualified far enough ahead of
    # the earliest disruption to actually be ready (lead-time constraint), pays
    # for its own one-time qualification cost out of the value it recovers on
    # exactly those periods? Evaluate every candidate's net insurance payoff and
    # only buy the option that is genuinely worth it -- reacting AFTER a
    # disruption starts is structurally too late (that is why this is a timing
    # decision, not just "diversify more").
    primary = 0
    best_base = suppliers[0]["tiers"][0][1]
    for i in range(N):
        if suppliers[i]["qualified"] and suppliers[i]["tiers"][0][1] < best_base:
            primary = i
            best_base = suppliers[i]["tiers"][0][1]

    grp0 = suppliers[primary]["group"]
    gap_periods = [t for t in range(1, T + 1) if (t, grp0) in disruptions]

    chosen = None  # (idx, start_period, covered_set)
    if gap_periods:
        best_net = 0.0
        for i in range(N):
            if i == primary or suppliers[i]["group"] == grp0:
                continue  # same correlation group as primary: worthless as insurance
            safe_gaps = [t for t in gap_periods if (t, suppliers[i]["group"]) not in disruptions]
            if not safe_gaps:
                continue
            start = max(1, min(safe_gaps) - suppliers[i]["lead"])
            avail_from = start + suppliers[i]["lead"]
            covered = [t for t in safe_gaps if t >= avail_from]
            if not covered:
                continue
            recovered = 0.0
            for t in covered:
                Dt = D[t - 1]
                delivered_value = Dt * V - tier_cost(suppliers[i], Dt)
                shortfall_value = -Dt * P
                recovered += delivered_value - shortfall_value
            net = recovered - suppliers[i]["qualcost"]
            if net > best_net:
                best_net = net
                chosen = (i, start, set(covered))

    covered_set = chosen[2] if chosen else set()

    orders = []
    for t in range(1, T + 1):
        Dt = D[t - 1]
        if Dt <= 0:
            continue
        if t in covered_set:
            orders.append((t, chosen[0], Dt))
        else:
            orders.append((t, primary, Dt))

    out = []
    if chosen:
        out.append("1")
        out.append(f"{chosen[0]} {chosen[1]}")
    else:
        out.append("0")
    out.append(str(len(orders)))
    for t, i, q in orders:
        out.append(f"{t} {i} {q}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
