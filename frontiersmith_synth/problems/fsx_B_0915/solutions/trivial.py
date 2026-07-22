# TIER: trivial
# Blind, throughput-limited: dispatch at most ONE request per tick (the
# oldest pending one) to the lowest-id idle vehicle, always via the
# preferred curb, direct route, never pools.
import sys, json

def main():
    view = json.load(sys.stdin)
    pending = view["pending"]
    idle = [v for v in view["vehicles"] if v["status"] == "idle"]
    if not pending or not idle:
        print(json.dumps({"assign": {}}))
        return
    r = min(pending, key=lambda r: (r["release_tick"], r["id"]))
    v = min(idle, key=lambda v: v["id"])
    stops = [{"action": "pickup", "request": r["id"], "curb": r["pickup_pref"]},
             {"action": "dropoff", "request": r["id"]}]
    print(json.dumps({"assign": {str(v["id"]): stops}}))

main()
