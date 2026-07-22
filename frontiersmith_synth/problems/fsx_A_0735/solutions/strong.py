# TIER: strong
# Bet-hedging portfolio policy, reconstructed fresh from `history` every call
# (this process has no memory of its own).
#
# Insight: a tip's current reading only tells you about ITS OWN position, so
# fully committing to today's best-looking tip (greedy) can never learn
# whether a currently-boring tip is hiding something much bigger deeper down.
# Conversely, spreading every step's budget evenly forever (uniform) never
# concentrates enough carbon on a genuinely good find to make the most of it.
#
# So: while no tip's reading clearly stands out above the field's typical
# (median) reading, spend the WHOLE step budget probing the single
# least-invested-so-far tip (round-robin full-budget exploration -- fast,
# systematic coverage, one tip at a time, not diluted across all of them).
# The moment some tip's reading clearly stands out (a genuine patch, not
# noise), it becomes the "leader": most of the budget (LEADER_FRAC) goes
# there to harvest it, while a guaranteed remainder keeps rotating through
# the other tips -- so a *second*, even better patch can still be found, and
# so the policy naturally drops a leader the instant its patch is spent
# (its reading falls back to the pack and it stops looking special).
import sys, json, math

inst = json.load(sys.stdin)
tips = inst["tips"]
B = inst["budget_step"]
history = inst.get("history", [])

MARGIN = 0.5          # how far above the field's median reading counts as "a find"
LEADER_FRAC = 0.75     # share of the budget committed to a confirmed leader

if not tips:
    print(json.dumps({"alloc": {}}))
    sys.exit(0)

ids = [t["id"] for t in tips]
sensed_now = {t["id"]: t["sensed"] for t in tips}

# Reconstruct cumulative carbon already invested in each tip from history.
invested = {i: 0.0 for i in ids}
for rec in history:
    for k, v in rec.get("alloc", {}).items():
        i = int(k)
        if i in invested:
            invested[i] += v

vals = sorted(sensed_now.values())
m = len(vals)
median = vals[m // 2] if m % 2 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
elevated = [i for i in ids if sensed_now[i] - median > MARGIN]

alloc = {str(i): 0.0 for i in ids}
if elevated:
    leader = max(elevated, key=lambda i: (sensed_now[i], -i))
    others = [i for i in ids if i != leader]
    if not others:
        alloc[str(leader)] = B
    else:
        explorer = min(others, key=lambda i: (invested[i], i))
        probe = (1.0 - LEADER_FRAC) * B
        alloc[str(explorer)] = probe
        alloc[str(leader)] += (B - probe)
else:
    explorer = min(ids, key=lambda i: (invested[i], i))
    alloc[str(explorer)] = B

print(json.dumps({"alloc": alloc}))
