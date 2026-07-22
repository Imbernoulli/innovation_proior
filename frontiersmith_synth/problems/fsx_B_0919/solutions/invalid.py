# TIER: invalid
# Dumps every job onto node 0 starting at its own arrival step, ignoring
# exclusivity entirely -- massively overlapping intervals on the same
# node. The checker must reject the whole test case (score 0) for this.
import sys, json

def main():
    inst = json.load(sys.stdin)
    jobs = inst["jobs"]
    sched = [{"id": j["id"], "node": 0, "start": j["arrival"]} for j in jobs]
    print(json.dumps({"schedule": sched}))

main()
