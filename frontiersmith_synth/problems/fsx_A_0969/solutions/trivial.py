# TIER: trivial
# Do nothing. Spend zero of the adaptive probe budget every round, then certify with the
# safest possible number: the published global cap m_cap. This is always sound (the grader's
# own recomputed D_earn from an unprobed/pilot-only history never exceeds m_cap) but useless.
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")

if phase == "probe":
    print(json.dumps({"probes": []}))
else:
    print(json.dumps({"bound": inst["m_cap"]}))
