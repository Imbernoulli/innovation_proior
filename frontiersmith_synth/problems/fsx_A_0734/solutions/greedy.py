# TIER: greedy
# "Set-and-forget profit-maximizing coupon." A textbook one-shot optimization: assume
# a generic, industry-average price sensitivity (beta_generic = 0.7) -- since the true
# per-segment elasticity is hidden, and this solution never touches the pilot data at
# all -- and solve for the single constant discount depth that maximizes the FIRST
# week's profit, treating that as if it applies every week:
#     profit(d) ~= (base_rate + beta_generic*d) * ((price-cost) - price*d)
# a downward parabola in d; take its vertex, clip to [0, max_discount], and apply that
# SAME constant depth every week for the whole year. This textbook recipe recomputes
# per-segment (using only public base_rate) but never adapts and never accounts for
# the fact that repeating any discount forever drags the segment's reference price
# down to match it, which erases the very demand bump the formula assumed would keep
# recurring -- for segments whose reference price adapts quickly this locks in a
# permanently thinner margin for most of the year.
import sys, json

BETA_GENERIC = 1.5

inst = json.load(sys.stdin)
price = inst["price"]
cost = inst["cost"]
dmax = inst["max_discount"]
n_weeks = inst["n_weeks"]
margin0 = price - cost

schedule = []
for seg in inst["segments"]:
    base = seg["base_rate"]
    # profit(d) = (base + b*d) * (margin0 - price*d)
    #           = base*margin0 + d*(b*margin0 - base*price) - b*price*d^2
    # vertex: d* = (b*margin0 - base*price) / (2*b*price)
    num = BETA_GENERIC * margin0 - base * price
    den = 2.0 * BETA_GENERIC * price
    d_star = num / den if den > 0 else 0.0
    if d_star < 0.0:
        d_star = 0.0
    if d_star > dmax:
        d_star = dmax
    schedule.append([d_star] * n_weeks)

print(json.dumps({"schedule": schedule}))
