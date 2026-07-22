# TIER: greedy
# The obvious first approach: reactive safety-stock rebalancing off the
# CURRENT bin's raw sensor reading, drawing from the regional relief hub.
# This program is invoked fresh once per bin, so to make consistent
# decisions it deterministically REPLAYS its own past decisions from step
# 0 up through the end of the CURRENT bin (using only `known_activity`,
# which covers exactly the bins causally available by now) -- an
# inexpensive re-derivation, not a lookahead: every replayed step only
# ever consults the bin index that step itself falls in, which is always
# <= the current epoch.
#
# The policy itself is naive: treat the CURRENT bin's cluster reading AS
# IF it always signals a typical, averaged bump in demand for that
# cluster's depots (no distinction between a fresh strike and ordinary
# ongoing dwell, no anticipation of the KNOWN cyclic advance to the NEXT
# cluster). Whenever a depot's simulated stock looks short RIGHT NOW,
# request a hub shipment sized to the observed gap, issued this very
# step. Because a shipment issued now cannot land for L more steps, this
# always misses the initial strike (the dominant source of damage) and
# only ever helps the smaller ongoing demand of an already-continuing
# dwell -- so on any instance where the storm's typical dwell time is
# shorter than L, essentially every reactive shipment lands after the
# real shortage has already cycled on to the next cluster.
import sys, json

inst = json.load(sys.stdin)
D, L = inst["D"], inst["L"]
clusters = inst["clusters"]
base_demand = inst["base_demand"]
capacity = inst["capacity"]
hub_capacity = inst["hub_capacity"]
shock_mean = inst["shock_mean"]
bin_width = inst["bin_width"]
known_activity = inst["known_activity"]
epoch_start, epoch_end = inst["epoch_start"], inst["epoch_end"]
HUB = D

BUMP = 0.3
ACT_THRESH = 0.35

stock = list(inst["initial_stock"])
pending = {}
my_shipments = []

for t in range(epoch_end):
    for i in range(D):
        stock[i] = min(capacity[i], stock[i] + 1.15 * base_demand[i])
    for (j, qty) in pending.pop(t, []):
        stock[j] = min(capacity[j], stock[j] + qty)

    b = t // bin_width
    demand_est = [base_demand[i] + (shock_mean * BUMP
                  if known_activity[b][clusters[i]] >= ACT_THRESH else 0.0) for i in range(D)]

    hub_left = hub_capacity
    order = sorted(range(D), key=lambda i: -(demand_est[i] - stock[i]))
    step_ships = []
    for i in order:
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
