# TIER: greedy
# Textbook recipe: "toll to equalize travel times" congestion pricing. Each round,
# tax every edge in proportion to how much its CURRENT latency exceeds the
# cheapest edge's latency right now, using only last round's realized load --
# a plausible first instinct, but purely REACTIVE: it damps whatever is
# congested at this instant instead of anticipating where the crowd's own
# best-response will send the flow next. The candidate self-simulates the
# stated dynamics (it knows the formulas from the statement) to produce the
# full T-round schedule in one shot.
import sys, json

KAPPA = 2.0


def l_edge(a, b, c, x):
    return a * x * x + b * x + c


def main():
    inst = json.load(sys.stdin)
    E, T, N, rho = inst["E"], inst["T"], inst["N"], inst["rho"]
    edges = [(e["a"], e["b"], e["c"]) for e in inst["edges"]]
    x = list(inst["x0"])

    tolls = []
    for _t in range(T):
        cur_cost = [l_edge(a, b, c, x[e]) for e, (a, b, c) in enumerate(edges)]
        cmin = min(cur_cost)
        toll_t = [KAPPA * max(0.0, cur_cost[e] - cmin) for e in range(E)]
        tolls.append(toll_t)

        cost = [cur_cost[e] + toll_t[e] for e in range(E)]
        inv = [1.0 / max(v, 1e-12) for v in cost]
        s = sum(inv)
        share = [v / s for v in inv]
        x = [(1.0 - rho) * x[e] + rho * N * share[e] for e in range(E)]

    print(json.dumps({"tolls": tolls}))


main()
