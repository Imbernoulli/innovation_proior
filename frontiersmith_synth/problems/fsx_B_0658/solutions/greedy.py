# TIER: greedy
# The obvious first idea: spread the extra sampling budget evenly across
# every region. Uses the whole budget (better than trivial) but completely
# ignores the pilot's noise signal AND region width -- on trap instances
# (a narrow region with a huge hidden noise spike) this wastes most of the
# budget on calm, well-behaved regions and under-serves the one region
# that actually needs it.
import sys, json

inst = json.load(sys.stdin)
regions = inst["regions"]
R = len(regions)
B = inst["budget"]
share = B / R
print(json.dumps({"alloc": [share] * R}))
