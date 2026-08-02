# TIER: greedy
# The obvious, individually-rational policy: every van plans its OWN charging in
# isolation. Since charging to full at the late stop p2 is always individually
# sufficient to finish the route, and charging earlier looks unnecessary from a
# single van's point of view, every van just tops up exactly what it consumed to
# reach p2, and skips the early stop p1 entirely. This is per-van threshold
# charging -- optimal for one van alone, but when the whole fleet does it at once,
# many vans' late stops share the SAME popular charger and their charging windows
# cluster: they queue for a scarce plug and later deliveries slip past deadline,
# even though each van's own decision was locally correct. This never looks at
# any other van's plan, so it cannot see (or avoid) that pileup.
import sys, json

inst = json.load(sys.stdin)

vans = []
for van in inst["vans"]:
    p2 = van["p2"]
    need_p2 = sum(leg["energy"] for leg in van["legs"][:p2])
    vans.append({"id": van["id"], "charge_at_p1": 0, "charge_at_p2": need_p2})

print(json.dumps({"vans": vans}))
