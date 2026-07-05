# TIER: trivial
# As-shipped curriculum: show the training examples in exactly the (shuffled)
# order they arrive, cycling 0,1,...,N-1,0,1,... up to the cap.  This is the
# evaluator's weak reference schedule, so it converges in q_base updates and
# scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_examples"]
cap = inst["cap"]

schedule = [i % N for i in range(cap)]
print(json.dumps({"schedule": schedule}))
