#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the caravan-guild
insurance-menu problem. Prints "... Ratio: <float in [0,1]>" on its own final line.
"""
import math
import sys

MAX_CONTRACTS = 6


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    agents = []
    for _ in range(n):
        p = float(toks[idx]); a = float(toks[idx + 1]); L = float(toks[idx + 2]); u = float(toks[idx + 3])
        idx += 4
        agents.append((p, a, L, u))
    return agents


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_menu(path, n_agents):
    try:
        with open(path, "r") as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")
        return None
    if not toks:
        fail("empty output")
    try:
        k = int(toks[0])
    except ValueError:
        fail("first token is not an integer K")
        return None
    if k < 0 or k > MAX_CONTRACTS:
        fail("K=%d out of range [0,%d]" % (k, MAX_CONTRACTS))
    need = 1 + 2 * k
    if len(toks) < need:
        fail("output truncated: expected %d tokens, got %d" % (need, len(toks)))
    if len(toks) > need:
        fail("output has trailing garbage beyond K contracts")
    menu = []
    for j in range(k):
        try:
            P = float(toks[1 + 2 * j])
            c = float(toks[2 + 2 * j])
        except ValueError:
            fail("contract %d is not numeric" % j)
            return None
        if not (math.isfinite(P) and math.isfinite(c)):
            fail("contract %d has a non-finite value" % j)
        if P < -1e-9:
            fail("contract %d has negative premium" % j)
        if c < -1e-9 or c > 1.0 + 1e-9:
            fail("contract %d has coverage outside [0,1]" % j)
        P = max(0.0, P)
        c = min(1.0, max(0.0, c))
        menu.append((P, c))
    return menu


def simulate(agents, menu):
    """Each agent picks the utility-maximizing contract, or exits (ties -> exit,
    then lowest contract index). Returns the guild's TRUE aggregate profit
    (computed from each agent's TRUE risk p_i, not any self-report)."""
    total = 0.0
    for (p, a, L, u) in agents:
        best_u = u
        best_j = -1
        for j, (P, c) in enumerate(menu):
            resid = (1.0 - c) * L
            util = -(P + p * resid) - a * p * (1.0 - p) * resid * resid
            if util > best_u + 1e-9:
                best_u = util
                best_j = j
        if best_j >= 0:
            P, c = menu[best_j]
            total += P - p * c * L
    return total


def sweep_optimal_single_price(agents):
    """Exact best flat single full-coverage price: sort candidate thresholds
    (-u_i) descending, sweep, prefix-sum the expected payout. O(n log n)."""
    cands = sorted(((-u, p * L) for (p, a, L, u) in agents), key=lambda t: -t[0])
    best_profit = 0.0
    prefix_payout = 0.0
    for k, (thr, payout) in enumerate(cands, 1):
        prefix_payout += payout
        profit = thr * k - prefix_payout
        if profit > best_profit:
            best_profit = profit
    return best_profit


def flat_fair_price_profit(agents):
    """Weak internal reference: ONE full-coverage contract priced at the raw
    (zero-markup) population-mean expected loss -- ignores participation and
    adverse selection entirely."""
    n = len(agents)
    if n == 0:
        return 0.0
    mean_pl = sum(p * L for (p, a, L, u) in agents) / n
    return simulate(agents, [(mean_pl, 1.0)])


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]
    agents = read_instance(in_path)
    n = len(agents)
    menu = read_menu(out_path, n)

    F = simulate(agents, menu)

    w_base = flat_fair_price_profit(agents)          # naive fair-price anchor -> 0.10
    w_ref = max(1e-6, sweep_optimal_single_price(agents))  # optimal flat-price anchor -> 0.80

    denom = max(1e-6, w_ref - w_base)
    r = 0.10 + 0.70 * (F - w_base) / denom
    r = max(0.0, min(1.0, r))

    print("agents=%d contracts=%d F=%.4f w_base=%.4f w_ref=%.4f" % (n, len(menu), F, w_base, w_ref))
    print("Ratio: %.6f" % r)


if __name__ == "__main__":
    main()
