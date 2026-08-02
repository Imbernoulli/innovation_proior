# TIER: greedy
# The obvious "use the freshest data" fix: reasoning that older observations are
# closer to steady state, pool only the most age-mature rows visible (ages within
# 3 of the max observed age) and average those as the long-run prediction. This
# still carries an un-removed fraction of the novelty spike whenever the decay
# time constant is comparable to (or longer than) the visible window, AND it is
# still contaminated by that day's common calendar wobble -- it never notices
# that OTHER cohorts, visible on the very same calendar days, could be used to
# cancel that wobble out. It over-predicts the long-run persistent lift.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.05"); sys.exit(0)
n = int(data[0])
vals = data[2:]
rows = []
for i in range(n):
    c = int(vals[5 * i]); s = int(vals[5 * i + 1]); t = int(vals[5 * i + 2])
    age = int(vals[5 * i + 3]); L = float(vals[5 * i + 4])
    rows.append((c, s, t, age, L))

max_age = max(r[3] for r in rows)
mature = [r[4] for r in rows if r[3] >= max_age - 3]
pred = sum(mature) / len(mature)
print("%.10g" % pred)
