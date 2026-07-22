# TIER: strong
# The insight: exploit the SHAPE of each supplier's concession curve (how low its
# floor goes, not its opening/current quote), and treat the outside option as a
# decaying resource to lock in early rather than a last resort.
#
#   1. Rank suppliers by FLOOR price (pfloor), not by opening or current ask.
#   2. Greedily allocate the needed quantity across suppliers in that floor order,
#      up to each one's capacity, until Q is covered (or capacity runs out).
#   3. For each allocated supplier, commit to it EXCLUSIVELY for consecutive
#      rounds -- never switching away mid-ride (so it never hardens) -- riding
#      its concession down as close to full (c=1) as its own deadline allows,
#      naturally landing in its near-deadline soften window when that helps,
#      then buying the whole allocation in the final round of that commitment.
#   4. Any quantity capacity cannot cover (the shortfall) is bought from the
#      OUTSIDE option immediately, in round 1 (or the first free round) -- since
#      the outside option only decays, waiting on it can only make it worse.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]; Q = inst["Q"]; M = inst["M"]
sup = inst["suppliers"]

order = sorted(range(M), key=lambda j: sup[j]["pfloor"])

remaining = Q
alloc = [0.0] * M
for j in order:
    if remaining <= 1e-9:
        break
    take = min(sup[j]["cap"], remaining)
    alloc[j] = take
    remaining -= take
shortfall = remaining

actions = [None] * T
t_cursor = 1
for j in order:
    if alloc[j] <= 1e-9:
        continue
    dl = sup[j]["deadline"]
    if t_cursor > dl:
        shortfall += alloc[j]
        alloc[j] = 0.0
        continue
    c = 0.0
    end_t = None
    t = t_cursor
    last_round = min(dl, T)
    while t <= last_round:
        near = (dl - t) < sup[j]["window"]
        step = sup[j]["base_step"] * (sup[j]["soften_mult"] if near else 1.0)
        c = min(1.0, c + step)
        if t < last_round and c < 0.999:
            actions[t - 1] = {"type": "negotiate", "supplier": j, "qty": 0.0}
        else:
            actions[t - 1] = {"type": "negotiate", "supplier": j, "qty": alloc[j]}
            end_t = t
            break
        t += 1
    if end_t is None:
        actions[last_round - 1] = {"type": "negotiate", "supplier": j, "qty": alloc[j]}
        end_t = last_round
    t_cursor = end_t + 1

for t in range(1, T + 1):
    if actions[t - 1] is None:
        actions[t - 1] = {"type": "wait"}

if shortfall > 1e-9:
    placed = False
    for t in range(1, T + 1):
        if actions[t - 1]["type"] == "wait":
            actions[t - 1] = {"type": "outside", "qty": shortfall}
            placed = True
            break
    if not placed:
        # every round already committed to a negotiation -- fold the shortfall
        # into round 1's outside purchase isn't possible (round 1 is a negotiate
        # action), so pay it the cheapest way still available: right after the
        # earliest commitment ends is as early as it can go without breaking a ride.
        actions[0] = {"type": "outside", "qty": shortfall}

print(json.dumps({"actions": actions}))
