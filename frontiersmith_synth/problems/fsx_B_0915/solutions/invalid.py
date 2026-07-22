# TIER: invalid
import sys, json

def main():
    view = json.load(sys.stdin)
    idle = [v for v in view["vehicles"] if v["status"] == "idle"]
    pending = view["pending"]
    if idle and pending:
        v = idle[0]
        r = pending[0]
        # deliberately bogus: dropoff a request never picked up
        print(json.dumps({"assign": {str(v["id"]): [{"action": "dropoff", "request": r["id"]}]}}))
    else:
        print(json.dumps({"assign": {}}))

main()
