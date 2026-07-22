# TIER: strong
# Start from the same reactive safety-stock lever the naive policy uses
# (so none of its captured value is lost), then add the genuine insight:
# use the CURRENT bin's per-cluster reading to identify which cluster is
# confidently active RIGHT NOW, then exploit the PUBLICLY KNOWN cyclic
# transition structure (storms always advance g -> (g+1) % G) to
# pre-stage a hub batch toward the NEXT cluster's depots immediately --
# paying the lead-time delay ahead of the eventual strike instead of
# reacting to it after the fact. This program is invoked fresh once per
# bin, so it deterministically REPLAYS its own past decisions from step 0
# through the end of the current bin using `known_activity` (which covers
# exactly the bins causally available by now -- every replayed step only
# ever consults the bin its own step falls in, never a future one, so
# this is re-derivation, not lookahead) to keep its internal stock
# estimate and "have I already committed to this transition" bookkeeping
# consistent across calls.
import sys, json

inst = json.load(sys.stdin)
D, L, G = inst["D"], inst["L"], inst["G"]
clusters = inst["clusters"]
base_demand = inst["base_demand"]
capacity = inst["capacity"]
hub_capacity = inst["hub_capacity"]
shock_mean = inst["shock_mean"]
bin_width = inst["bin_width"]
known_activity = inst["known_activity"]
epoch_start, epoch_end = inst["epoch_start"], inst["epoch_end"]
HUB = D

depots_by_cluster = [[i for i in range(D) if clusters[i] == g] for g in range(G)]

ACTIVE_THRESH = 0.30
DOMINANCE_MARGIN = 0.05
PER_TARGET_MULT = 0.4
BUMP = 0.4

stock = list(inst["initial_stock"])
pending = {}
my_shipments = []
last_target = None
last_bin = -1

for t in range(epoch_end):
    for i in range(D):
        stock[i] = min(capacity[i], stock[i] + 1.15 * base_demand[i])
    for (j, qty) in pending.pop(t, []):
        stock[j] = min(capacity[j], stock[j] + qty)

    b = t // bin_width
    step_ships = []
    hub_left = hub_capacity

    # --- anticipatory layer: predict the NEXT cluster from the current bin ---
    scores = known_activity[b]
    order = sorted(range(G), key=lambda g: -scores[g])
    best_g, second_g = order[0], order[1]
    conf, runner_up = scores[best_g], scores[second_g]
    confident = conf >= ACTIVE_THRESH and (conf - runner_up) >= DOMINANCE_MARGIN

    if confident and b != last_bin:
        target_g = (best_g + 1) % G
        if target_g != last_target:
            per_target = shock_mean * PER_TARGET_MULT
            for tg in depots_by_cluster[target_g]:
                send = min(per_target, hub_left)
                if send > 1e-9:
                    step_ships.append([t, HUB, tg, send])
                    hub_left -= send
                    pending.setdefault(t + L, []).append((tg, send))
            last_target = target_g
        last_bin = b
    elif not confident:
        last_target = None

    # --- reactive layer: same immediate-bin backstop the naive policy uses ---
    demand_est = [base_demand[i] + (shock_mean * BUMP
                  if known_activity[b][clusters[i]] >= ACTIVE_THRESH else 0.0) for i in range(D)]
    reactive_order = sorted(range(D), key=lambda i: -(demand_est[i] - stock[i]))
    for i in reactive_order:
        gap = demand_est[i] - stock[i]
        if gap <= 0 or hub_left <= 1e-9:
            continue
        send = min(gap, hub_left)
        step_ships.append([t, HUB, i, send])
        hub_left -= send
        pending.setdefault(t + L, []).append((i, send))

    for i in range(D):
        served = min(stock[i], demand_est[i])
        stock[i] -= served

    if t >= epoch_start:
        my_shipments.extend(step_ships)

print(json.dumps({"shipments": my_shipments}))
