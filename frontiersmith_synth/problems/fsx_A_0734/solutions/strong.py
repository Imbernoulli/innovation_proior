# TIER: strong
# Genuine insight: IDENTIFY each segment's hidden memory constant (alpha) and price
# sensitivity (beta) from the public pilot data in closed form, decide whether the
# segment is even worth discounting, and if so RATION the discount as short pulses
# (a burst, then full price to let the reference price recover) instead of a constant
# giveaway -- searching the pulse depth/period that a *local* simulation (using the
# identified parameters) says is best, rather than assuming a one-shot formula holds
# forever.
#
# Identification (closed form, from the stated reference-price law):
#   The pilot applies a shock discount in week 0 (reference price is still at full
#   price P at the start of week 0), then holds full price for the remaining pilot
#   weeks. Let frac_t = pilot_demand[t]/size.
#     beta_hat  = (frac_0 - base_rate) / shock_depth          (week-0 gap = P*shock)
#   During the full-price recovery weeks the *erosion* erosion_t := base_rate-frac_t
#   is proportional to (P - ref_t), and (P - ref_t) decays geometrically with ratio
#   (1 - alpha) each week the price is held at P, so:
#     alpha_hat = 1 - mean_t[ erosion_{t+1} / erosion_t ]     (recovery weeks)
#
# Scheduling: if the identified one-shot marginal value of discounting at d=0 is
# <= 0, never discount this segment. Otherwise grid-search a periodic pulse (depth,
# period) using the segment's OWN identified alpha_hat/beta_hat in a local replay of
# the exact reference-price law, and commit the best pulse pattern for the year.
import sys, json

inst = json.load(sys.stdin)
price = inst["price"]
cost = inst["cost"]
dmax = inst["max_discount"]
n_weeks = inst["n_weeks"]
pilot_depths = inst["pilot_depths"]
margin0 = price - cost


STRUCT_FRAC = 0.25  # stated modeling constant: share of elasticity that is permanent


def local_simulate(depths, base_rate, size, alpha, beta):
    ref = price
    total = 0.0
    for d in depths:
        p = price * (1.0 - d)
        struct_gap = price - p
        ref_gap = ref - p
        frac = (base_rate + beta * STRUCT_FRAC * (struct_gap / price)
                + beta * (1.0 - STRUCT_FRAC) * (ref_gap / price))
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        total += size * frac * (p - cost)
        ref = alpha * p + (1.0 - alpha) * ref
    return total


PULSE_DEPTHS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
PULSE_PERIODS = [1, 2, 3, 4, 6, 8, 13, 26, 52]

schedule = []
for seg in inst["segments"]:
    base = seg["base_rate"]
    size = seg["size"]
    pilot_demand = seg["pilot_demand"]
    fracs = [u / size for u in pilot_demand]

    shock_depth = pilot_depths[0]
    beta_hat = (fracs[0] - base) / shock_depth if shock_depth > 1e-9 else 0.5
    if beta_hat < 0.02:
        beta_hat = 0.02

    erosions = [base - f for f in fracs[1:]]  # recovery-week erosion, weeks 1..K-1
    ratios = []
    for t in range(len(erosions) - 1):
        e0, e1 = erosions[t], erosions[t + 1]
        if e0 > 1e-6:
            ratios.append(e1 / e0)
    if ratios:
        alpha_hat = 1.0 - sum(ratios) / len(ratios)
    else:
        alpha_hat = 0.3
    alpha_hat = max(0.02, min(0.98, alpha_hat))

    marg = beta_hat * margin0 - base * price
    if marg <= 0.0:
        schedule.append([0.0] * n_weeks)
        continue

    best_profit = None
    best_sched = [0.0] * n_weeks
    for dp in PULSE_DEPTHS:
        if dp > dmax + 1e-9:
            continue
        for T in PULSE_PERIODS:
            depths = [dp if (w % T == 0) else 0.0 for w in range(n_weeks)]
            prof = local_simulate(depths, base, size, alpha_hat, beta_hat)
            if best_profit is None or prof > best_profit:
                best_profit = prof
                best_sched = depths
    # also compare against never-discounting this segment
    never = local_simulate([0.0] * n_weeks, base, size, alpha_hat, beta_hat)
    if best_profit is None or never > best_profit:
        best_sched = [0.0] * n_weeks
    schedule.append(best_sched)

print(json.dumps({"schedule": schedule}))
