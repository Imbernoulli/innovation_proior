# TIER: greedy
# The obvious recipe: every step, dump the ENTIRE budget into whichever
# currently ACTIVE tip has the highest currently-SENSED concentration
# (ties -> lowest tip id). This is a pure commitment / no-portfolio policy:
# once a tip's local reading looks best it gets fully exploited, so a big
# patch sitting deeper on a currently-boring tip is never discovered because
# that tip is never advanced far enough to reach it.
import sys, json

inst = json.load(sys.stdin)
tips = inst["tips"]
B = inst["budget_step"]

if not tips:
    print(json.dumps({"alloc": {}}))
else:
    best = max(tips, key=lambda t: (t["sensed"], -t["id"]))
    alloc = {str(t["id"]): 0.0 for t in tips}
    alloc[str(best["id"])] = B
    print(json.dumps({"alloc": alloc}))
