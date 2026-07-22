# TIER: trivial
# Do nothing: keep every key on its current shard. This is exactly the evaluator's
# do-nothing reference, so it scores 0.1 on every instance no matter how skewed the
# initial placement is.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"assign": list(inst["shard0"])}))
