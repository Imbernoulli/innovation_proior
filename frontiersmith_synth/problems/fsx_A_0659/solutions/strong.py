# TIER: strong
# Insight: read the instance's OWN eta_c/eta_d/aging_coeff/degradation_price
# to compute the true marginal breakeven for a round trip (efficiency tax
# AND the quadratic depth-aging cost), decompose the full seeded price
# series into its alternating local valleys/peaks (a classic buy-low /
# sell-high leg exchange-decomposition -- the whole series is public, no
# online constraint), keep ONLY the legs whose spread clears that breakeven
# with margin, and IGNORE everything else (the noise). For each surviving
# leg, ladder the trade across a small window around its extremum (several
# shallower steps instead of one deep dump) -- charging is aging-free so we
# usually charge freely during a buy window, but discharging pays the
# quadratic depth penalty, so spreading a sell across a couple of steps
# (when the power limit does not already force that) strictly lowers the
# aging bill for the same captured spread. Finally, since the WHOLE series
# is visible up front, rank every surviving leg's spread quality globally
# and cap how much capacity a merely-decent leg may claim -- so an early
# mediocre spread cannot fill the battery and starve the one truly large
# seeded swing that shows up later. This reserves capacity headroom for
# the handful of genuinely large seeded swings instead of bleeding it away
# on marginal noise or over-committing to the first decent spread.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
prices = inst["prices"]
cap0 = float(inst["capacity0"])
pmax = float(inst["power_max"])
soc0 = float(inst["soc0"])
eta_c = float(inst["eta_c"])
eta_d = float(inst["eta_d"])
aging_coeff = float(inst["aging_coeff"])
degradation_price = float(inst["degradation_price"])


def leg_worth(buy_p, sell_p):
    # Representative single-unit round trip at depth pmax (conservative: a
    # deeper dump is even less favorable, so this is a safe screening test).
    x = pmax
    d = x * eta_c
    cost = x * buy_p
    dod = d / cap0 if cap0 > 1e-9 else 0.0
    fade = aging_coeff * cap0 * dod * dod
    aging_cost = degradation_price * fade
    revenue = d * eta_d * sell_p
    return (revenue - cost - aging_cost) > 0.05 * max(cost, 1e-6)


# --- alternating local-extrema decomposition of the full price series ---
extrema = [0]
direction = 0
for t in range(1, T):
    if prices[t] > prices[t - 1]:
        d = 1
    elif prices[t] < prices[t - 1]:
        d = -1
    else:
        d = 0
    if d == 0:
        continue
    if direction == 0:
        direction = d
    elif d != direction:
        extrema.append(t - 1)
        direction = d
if extrema[-1] != T - 1:
    extrema.append(T - 1)

# A raw local-extrema scan also flags tiny noise wiggles as "legs" -- a
# wiggle sitting on a low price base can even clear the *ratio* breakeven
# test while being economically irrelevant (a few-dollar zigzag, not a
# seeded swing). Screen those out with a robust noise-scale estimate (the
# median step-to-step price move) before the economic test: keep only legs
# whose absolute gain is a clear multiple of that scale.
diffs = sorted(abs(prices[t] - prices[t - 1]) for t in range(1, T))
noise_scale = diffs[len(diffs) // 2] if diffs else 0.0
min_gain = max(6.0 * noise_scale, 8.0)

# --- keep only RISE legs (valley -> peak) that are both a clear (non-noise)
#     gain AND clear the efficiency+aging breakeven test ---
legs = []
for i in range(len(extrema) - 1):
    a, b = extrema[i], extrema[i + 1]
    if prices[b] - prices[a] >= min_gain and leg_worth(prices[a], prices[b]):
        legs.append((a, b))

# --- exchange argument: two adjacent surviving legs separated by a dip
#     (sell at the first peak, buy back at the next valley) are only worth
#     keeping SEPARATE if that intermediate cash-out clears the SAME
#     breakeven test -- otherwise the round-trip friction of exiting and
#     re-entering costs more than just holding through the dip, so merge
#     the two legs into one bigger valley-to-peak leg instead.
stack = []
for leg in legs:
    while stack and not leg_worth(prices[leg[0]], prices[stack[-1][1]]):
        prev = stack.pop()
        leg = (prev[0], leg[1])
    stack.append(leg)
legs = stack

# --- global capacity budgeting: with full lookahead over the WHOLE series,
#     rank every surviving leg by its achieved spread quality
#     (eta_c*eta_d*sell / buy). A mediocre early leg that greedily fills to
#     full capacity leaves nothing in reserve for a much better swing later
#     -- so only legs within reach of the single BEST quality in the whole
#     instance get an unrestricted buy budget; clearly inferior legs get a
#     capped budget, preserving most of the battery's capacity for the
#     handful of genuinely large seeded swings instead of spending it on
#     the first decent-looking spread that comes along.
def quality(v, p):
    bp = prices[v]
    return (eta_c * eta_d * prices[p]) / bp if bp > 1e-9 else 0.0

quals = [quality(v, p) for (v, p) in legs]
best_q = max(quals) if quals else 0.0
budgets = []
for q in quals:
    budgets.append(cap0 if (best_q <= 1e-9 or q >= 0.85 * best_q) else 0.35 * cap0)

# --- claim a small execution window around each surviving leg's valley
#     (buy) and peak (sell). Windows extend ONLY away from the leg's own
#     interior (buy window looks BACKWARD from the valley, sell window
#     looks FORWARD from the peak) so a short valley-to-peak gap can never
#     let a buy window swallow the peak's own index (or vice versa) --
#     that directional constraint is what keeps the ladder from ever
#     trading in the wrong direction near a short leg.
#     Also require the ACTUAL price at each claimed step to stay close to
#     the extremum price (within the noise scale) -- guards against a
#     window's backward/forward reach landing on an unrelated price level.
R = 2
tol = max(3.0 * noise_scale, 4.0)
claimed = [None] * T         # "buy" / "sell" per step
claimed_leg = [None] * T     # leg index owning a "buy" step (for its budget)
order = sorted(range(len(legs)), key=lambda i: legs[i][0])
for i in order:
    v, p = legs[i]
    for t in range(v, v - R - 1, -1):
        if 0 <= t < T and claimed[t] is None and prices[t] <= prices[v] + tol:
            claimed[t] = "buy"
            claimed_leg[t] = i
    for t in range(p, p + R + 1):
        if 0 <= t < T and claimed[t] is None and prices[t] >= prices[p] - tol:
            claimed[t] = "sell"

# --- forward simulate: charge (up to that leg's budget) during buy
#     windows, drain during sell windows, hold everywhere else ---
soc = soc0
cap = cap0
bought = [0.0] * len(legs)
actions = [0.0] * T
for t in range(T):
    kind = claimed[t]
    if kind == "buy":
        li = claimed_leg[t]
        remaining_budget = max(0.0, budgets[li] - bought[li])
        room = (cap - soc) / eta_c if eta_c > 1e-9 else 0.0
        x = min(pmax, max(0.0, room), remaining_budget)
        if x > 1e-9:
            actions[t] = x
            soc = min(soc + x * eta_c, cap)
            bought[li] += x * eta_c
    elif kind == "sell":
        d = min(pmax, soc)
        if d > 1e-9:
            actions[t] = -d
            dod = d / cap if cap > 1e-9 else 0.0
            fade = aging_coeff * cap0 * dod * dod
            cap = max(cap - fade, 0.05 * cap0)
            soc -= d

print(json.dumps({"actions": actions}))
