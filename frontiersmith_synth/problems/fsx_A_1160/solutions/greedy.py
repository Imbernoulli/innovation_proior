# TIER: greedy
# The obvious first algorithm: always use corridor 0 (the shortest, cheapest
# stress factor), and correctly simulate its OWN fatigue state day by day so
# it always loads exactly that day's TRUE safe capacity -- it never
# overloads and never cracks. This "gets the physics right" for a single
# corridor.
#
# The trap: because it never rests corridor 0 and never spreads load to the
# other three, corridor 0's fatigue clock never gets a chance to fall behind
# its thickness clock -- crossing at 100% of capacity every single day keeps
# fatigue pinned close to thickness itself, so the corridor's *effective*
# thickness (and hence deliverable mass) stays capped near a low ceiling for
# the entire season. Meanwhile corridors 1-3 -- untouched, so their fatigue
# is always 0 -- grow thick and are never exploited, especially during
# whatever stretch of the season turns out coldest. Treating "shortest
# corridor" as a fixed path choice instead of a spacetime budget costs most
# of the season's real capacity.
import sys, json

inst = json.load(sys.stdin)
n = inst["n_days"]
m = inst["mechanics"]
route = inst["routes"][0]
temps = inst["temps"]
freeze = inst["freeze_point"]

h = route["h0"]
fatigue = 0.0
routes_out = [0] * n
masses_out = [0.0] * n

for d in range(n):
    eff = max(h - fatigue, 0.0)
    ms = m["stress_limit"] * eff * eff / route["length_factor"]
    mass = ms  # load exactly today's true safe capacity, no margin
    masses_out[d] = mass

    if mass > 0.0:
        denom = m["stress_limit"] * max(eff, 1e-6) ** 2
        stress_frac = (mass * route["length_factor"]) / denom
        add = m["fatigue_gain_k"] * stress_frac * h
    else:
        add = 0.0
    fatigue = m["fatigue_decay"] * fatigue + add

    T = temps[d]
    g = route["growth_rate"] * max(0.0, freeze - T) - route["thaw_rate"] * max(0.0, T - freeze)
    h = max(0.0, h + g)

print(json.dumps({"routes": routes_out, "masses": masses_out}))
