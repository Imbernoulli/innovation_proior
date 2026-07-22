# TIER: strong
"""Spectral leverage + adaptive paced threshold -- genuinely online (this program
is invoked once per truck; it never sees any truck after the current one).

Insight #1 (fiedler-sketching): instead of recomputing the exact lambda_2 of a
hypothetical purchase (the myopic-greedy recipe), take a cheap Fiedler-vector
sketch of the network built SO FAR and use the classical rank-1 perturbation
estimate that adding edge (u, v) moves lambda_2 by roughly (x_u - x_v)^2 per
unit weight, where x is the Fiedler vector. Cables whose endpoints already sit
on the same side of the sketch (same cluster) barely move x -- these are the
"decoy" cables that myopic-greedy is fooled by. Cables crossing the sketch's
bottleneck have large (x_u - x_v)^2 and are the ones worth pacing for.

Insight #2 (threshold pacing / online-selection-under-budget): we cannot look
ahead, so we spend the first slice of the manifest purely OBSERVING leverage
values (buying nothing) to build a private sample of what "normal" leverage
looks like here. After that we track, using ONLY: (a) our own memory of past
observations (round-tripped through the private "state" JSON blob -- there is
no other memory across calls), and (b) the truck count/budget accounting the
evaluator hands us this call, an adaptive acceptance PERCENTILE: how large a
top-slice of the leverage distribution we can afford to keep buying from is
roughly (money left / average cost so far) trucks out of the trucks that are
still coming. When that ratio is small, only the truly best-looking cables
clear the bar; as it grows (or as the manifest runs low) the bar relaxes so
unspent money still gets used before the last truck leaves."""
import sys, json
import numpy as np

WARM_FRAC = 0.35
SAFETY = 0.7


def fiedler_vec(n, edges):
    L = np.zeros((n, n))
    for u, v, w in edges:
        L[u, u] += w; L[v, v] += w; L[u, v] -= w; L[v, u] -= w
    w, V = np.linalg.eigh(L)
    idx = int(np.argsort(w)[1])
    return V[:, idx]


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]; m = inst["m"]; t = inst["t"]
    remaining = inst["remaining"]
    u, v, cost = inst["u"], inst["v"], inst["cost"]
    state = inst.get("state") or {}
    seen_lev = list(state.get("seen_lev", []))
    seen_cost = list(state.get("seen_cost", []))

    edges = [(a, b, 1) for a, b in inst["backbone"]] + [(a, b, 1) for a, b in inst["accepted"]]
    x = fiedler_vec(n, edges)
    lev = float((x[u] - x[v]) ** 2) / max(cost, 1e-9)

    min_seen = max(3, int(WARM_FRAC * m))
    remaining_count = m - t

    if len(seen_lev) < min_seen:
        accept = False
    else:
        avg_cost = (sum(seen_cost) / len(seen_cost)) if seen_cost else float(cost)
        expected_afford = remaining / max(avg_cost, 1e-9)
        frac = min(1.0, SAFETY * expected_afford / max(remaining_count, 1))
        if frac >= 0.999:
            accept = cost <= remaining + 1e-9
        else:
            s = sorted(seen_lev)
            idx = min(len(s) - 1, int((1 - frac) * (len(s) - 1)))
            tau = s[idx]
            accept = lev >= tau and cost <= remaining + 1e-9

    seen_lev.append(lev)
    seen_cost.append(float(cost))
    new_state = {"seen_lev": seen_lev, "seen_cost": seen_cost}

    print(json.dumps({"accept": accept, "state": new_state}))


if __name__ == "__main__":
    main()
