# TIER: greedy
# The obvious first approach: a classic secretary-style sample-then-threshold
# rule applied directly to the RAW interview score, exactly as if "higher
# score means higher value" always held. Observe an initial warm-up sample,
# use it to calibrate an acceptance quantile aimed at roughly filling K
# slots over the rest of the stream, then accept every candidate clearing
# that bar until slots run out. Never touches the recall window (a passed
# candidate is treated as gone for good), never adjusts for the disclosed
# drift term (no reason to save slots for later), and -- critically -- never
# questions whether "higher score = higher value" could be false for this
# instance. On every Fading-regime instance this chases exactly the wrong
# tail of the score distribution.
import sys, json

inst = json.load(sys.stdin)
N, K = inst["N"], inst["K"]
scores = inst["score"]

m = max(5, round(0.25 * N))
warm = sorted(scores[:m])
target_frac = min(0.95, K / max(1, N - m))
idx = min(len(warm) - 1, max(0, int(round((1.0 - target_frac) * (len(warm) - 1)))))
thresh = warm[idx]

actions = [0] * N
hired = 0
for i in range(m, N):
    if hired >= K:
        break
    if scores[i] >= thresh:
        actions[i] = 1
        hired += 1

print(json.dumps({"actions": actions, "recalls": []}))
