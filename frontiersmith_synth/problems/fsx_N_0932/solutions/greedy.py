# TIER: greedy
# The obvious recipe: one-step lookahead, spread evenly. Each round, simulate ONE
# concession step for every still-active supplier (as if negotiating it this
# round), compare those resulting asks plus the current outside price, and press
# whichever is cheapest RIGHT NOW -- negotiating it and buying a modest, evenly
# spread lot (Q/T, a natural "don't put all your eggs in one round" sizing;
# the final round instead clears whatever remains, so the need is always met).
# This never sacrifices a round on a currently-worse supplier to find out
# whether its floor/step shape is actually better, and it only reaches for the
# (decaying) outside option once suppliers stop being the cheapest choice --
# i.e. reactively, not as a planned early lock for a known shortfall.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]; Q = inst["Q"]; M = inst["M"]
sup = inst["suppliers"]
outside0 = inst["outside0"]; growth = inst["outside_growth"]
lot = Q / T


def ask(s, c):
    return s["pfloor"] + (s["p0"] - s["pfloor"]) * (1.0 - c)


def outside_price(t):
    return outside0 * (growth ** (t - 1))


c = [0.0] * M
cap_rem = [s["cap"] for s in sup]
remaining = Q
actions = []

for t in range(1, T + 1):
    if remaining <= 1e-9:
        break
    best_ask = None
    best_j = None
    for j in range(M):
        if t > sup[j]["deadline"] or cap_rem[j] <= 1e-9:
            continue
        near = (sup[j]["deadline"] - t) < sup[j]["window"]
        step = sup[j]["base_step"] * (sup[j]["soften_mult"] if near else 1.0)
        c_try = min(1.0, c[j] + step)
        a = ask(sup[j], c_try)
        if best_ask is None or a < best_ask:
            best_ask = a
            best_j = j
    out_p = outside_price(t)
    take = remaining if t == T else min(remaining, lot)
    if best_j is None or out_p <= best_ask:
        actions.append({"type": "outside", "qty": take})
        remaining -= take
    else:
        q = min(take, cap_rem[best_j])
        actions.append({"type": "negotiate", "supplier": best_j, "qty": q})
        near = (sup[best_j]["deadline"] - t) < sup[best_j]["window"]
        step = sup[best_j]["base_step"] * (sup[best_j]["soften_mult"] if near else 1.0)
        c[best_j] = min(1.0, c[best_j] + step)
        cap_rem[best_j] -= q
        remaining -= q

print(json.dumps({"actions": actions}))
