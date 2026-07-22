# TIER: trivial
# Pure round-robin across ALL 2*S cars by arrival index -- no floor
# information used at all, not even a fixed midpoint split. Every car gets a
# fair, even share of calls, but WHICH shaft/role serves a call has nothing
# to do with where the call actually is. Parks every lower car at the
# bottom, every upper car at the top. This is the "do the simplest possible
# fair thing" construction and is also what the evaluator uses internally as
# its own weak reference (so this solution always reproduces the checker's
# baseline exactly).
import sys, json

inst = json.load(sys.stdin)
F, S = inst["F"], inst["S"]
calls = inst["calls"]
ncars = 2 * S

assign = [None] * len(calls)
for i, c in enumerate(calls):
    car = i % ncars
    assign[c["id"]] = [car // 2, car % 2]

park = []
for s in range(S):
    park.append(0)        # lower car parks at the bottom
    park.append(F - 1)    # upper car parks at the top

print(json.dumps({"assign": assign, "park": park}))
