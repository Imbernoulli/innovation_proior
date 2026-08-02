# TIER: greedy
# The obvious first strategy: "image the highest-value target next." Sort ALL
# targets by raw value descending (ties by id) and visit them in that order,
# single pass 1 only, never touching pass 2. This ignores slew cost entirely (a
# rich, far-off target still gets visited first, burning the whole pass on the
# trip there) and ignores the cloud forecast entirely (a target with a 90% chance
# of being clouded out gets imaged anyway, for a good chance of zero return) and
# never defers anything to the safer/second look. It reliably beats doing nothing,
# but is the trap this problem is built to punish.
import sys, json

inst = json.load(sys.stdin)
targets = sorted(inst["targets"], key=lambda t: (-t["value"], t["id"]))
ids = [t["id"] for t in targets]
print(json.dumps({"pass1": ids, "pass2": []}))
