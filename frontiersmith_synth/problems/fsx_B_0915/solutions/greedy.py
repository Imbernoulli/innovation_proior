# TIER: greedy
# The obvious first attempt: NEAREST-VEHICLE INSTANT DISPATCH, processed one
# request at a time in a single pass (oldest request first -- no global
# reoptimization across requests). For each pending request, in turn, send
# the nearest still-idle vehicle straight at the request's PREFERRED curb
# via a direct single-rider route -- never an alternate curb, never
# pooling, and no awareness that another vehicle might be converging on
# the same curb the same tick.
import sys, json

def main():
    view = json.load(sys.stdin)
    pending = sorted(view["pending"], key=lambda r: (r["release_tick"], r["id"]))
    idle_avail = [v for v in view["vehicles"] if v["status"] == "idle"]
    assign = {}
    for r in pending:
        if not idle_avail:
            break
        v = min(idle_avail, key=lambda v: (abs(v["pos"] - r["pickup_pref"]), v["id"]))
        stops = [{"action": "pickup", "request": r["id"], "curb": r["pickup_pref"]},
                  {"action": "dropoff", "request": r["id"]}]
        assign[str(v["id"])] = stops
        idle_avail = [x for x in idle_avail if x["id"] != v["id"]]
    print(json.dumps({"assign": assign}))

main()
