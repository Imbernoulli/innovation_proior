# TIER: strong
"""Strong candidate: exploit the storage memory, the stage sensitivity and the
tariff schedule TOGETHER.

Insight: simulate the field's own true two-layer water balance (it is fully
determined by the public rain/et/stage sequence and the physical params, so a
solver can replay it exactly). Whenever that simulation shows the FIRST future
day the root-zone would drop below its stage's comfort threshold, patch it by
adding a small chunk of water on the CHEAPEST still-available day at or before
that violation -- never later, because by the time the shortfall is visible it
is too late (the surface-to-root transfer is not instant: that is the field's
"memory"). Because the search always uses the fully-simulated forward
trajectory, it automatically:
  - never buys water immediately before a big forecasted rain event (the
    projected trajectory already accounts for that rain and shows no
    violation, so nothing is purchased there -- avoiding the surface-overflow
    waste a fixed-target rule falls into);
  - starts charging the root-zone reservoir MANY days before flowering if the
    percolation rate is slow, because the repair naturally reaches back to
    early, cheap days once nearby ones are exhausted;
  - shifts purchases to cheap tariff days even when the eventual need is a
    tariff SPIKE during flowering, banking water in the deep reservoir ahead
    of time.
This is not "greedy plus more iterations": it is a reformulation of an
open-loop schedule as a repeated cheapest-repair over the fully simulated
state trajectory -- an exchange-argument style construction, not a fixed rule.
"""
import sys, json


def simulate_first_violation(T, Cs, Cr, alpha, S0, R0, rain, et, stage, smin, irr):
    S, R = S0, R0
    for t in range(T):
        S = S + rain[t] + irr[t]
        if S > Cs:
            S = Cs
        perc = alpha * S
        room = max(0.0, Cr - R)
        pa = min(perc, room)
        S -= perc
        if S < 0.0:
            S = 0.0
        R += pa
        theta = R / Cr
        need = smin[stage[t]]
        if theta < need - 1e-6:
            return t
        R -= min(R, et[t])
    return None


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    p = inst["params"]
    Cs, Cr, alpha = p["Cs"], p["Cr"], p["alpha"]
    S0, R0 = p["S0"], p["R0"]
    maxirr = p["max_irrig_per_day"]
    smin = p["stage_theta_min"]
    rain, et, tariff, stage = inst["rain"], inst["et"], inst["tariff"], inst["stage"]

    irr = [0.0] * T
    chunk = max(0.5, maxirr / 24.0)
    max_iter = 8000

    for _ in range(max_iter):
        d = simulate_first_violation(T, Cs, Cr, alpha, S0, R0, rain, et, stage, smin, irr)
        if d is None:
            break
        best_k, best_price = None, None
        for k in range(0, d + 1):
            if irr[k] < maxirr - 1e-9:
                if best_price is None or tariff[k] < best_price:
                    best_price, best_k = tariff[k], k
        if best_k is None:
            break
        irr[best_k] = min(maxirr, irr[best_k] + chunk)

    print(json.dumps({"irrig": irr}))


if __name__ == "__main__":
    main()
