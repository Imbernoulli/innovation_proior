# TIER: strong
# Insight: detect the mid-stream drift, exponentially deweight the pre-drift
# counts, and reserve tail slots so the LATE, ACCELERATING risers out-rank the
# stale pre-drift leaders and survive in the bounded summary.
#   1. change-point detection: find where the early-dominant token set loses its
#      grip -> cp.
#   2. weighted-sketch-reallocation: score each token by a recency-weighted
#      count with occurrences before cp hard-deweighted (exp decay in position).
#   3. tail-reserve: set aside a few slots for freshly-emerging, up-trending
#      tokens (rising in the last slice) that the weighted rank alone would miss.
import sys, json, math
from collections import Counter, defaultdict

inst = json.load(sys.stdin)
K = inst["K"]; stream = inst["stream"]
n = len(stream)
if n == 0:
    print(json.dumps({"keep": []})); sys.exit(0)

# ---- occurrence positions ----
pos = defaultdict(list)
for i, t in enumerate(stream):
    pos[t].append(i)

# ---- 1. change-point: split into blocks; the "early leaders" are the top
# tokens of the first blocks. cp = first block where their combined share
# collapses below half of its early level. ----
B = 24
bl = max(1, n // B)
blocks = [stream[i * bl:(i + 1) * bl] for i in range((n + bl - 1) // bl)]
early = Counter()
for blk in blocks[:3]:
    early.update(blk)
leaders = {t for t, _ in early.most_common(max(4, K))}
share = []
for blk in blocks:
    if not blk:
        share.append(0.0); continue
    share.append(sum(1 for t in blk if t in leaders) / len(blk))
base = max(share[:3]) if share[:3] else 1.0
cp = 0
for bi in range(len(share)):
    if base > 0 and share[bi] < 0.5 * base:
        cp = bi * bl
        break
if cp == 0:
    cp = n // 3

# ---- 2. recency-weighted count with pre-cp hard deweight ----
lam = 4.5
DEN = float(max(1, n - 1))
w = {}
for t, ps in pos.items():
    s = 0.0
    for p in ps:
        rec = math.exp(-lam * (n - 1 - p) / DEN)     # recent -> ~1, old -> ~0
        if p < cp:
            rec *= 0.08                               # deweight pre-drift counts
        s += rec
    w[t] = s

# ---- 3. tail-reserve: tokens rising in the last 12% vs the previous 12% ----
tail = int(0.12 * n)
a_lo, a_hi = n - tail, n
b_lo, b_hi = n - 2 * tail, n - tail
rising = []
for t, ps in pos.items():
    recent = sum(1 for p in ps if a_lo <= p < a_hi)
    prev = sum(1 for p in ps if b_lo <= p < b_hi)
    if recent >= 2 and recent > prev:
        rising.append((recent - prev, recent, t))
rising.sort(key=lambda z: (-z[0], -z[1], z[2]))

reserve = 3
main_k = K - reserve
ranked = sorted(w, key=lambda t: (-w[t], t))
keep = []
seen = set()
for t in ranked:
    if len(keep) >= main_k:
        break
    keep.append(t); seen.add(t)
for _, _, t in rising:
    if len(keep) >= K:
        break
    if t not in seen:
        keep.append(t); seen.add(t)
for t in ranked:                        # backfill if reserve found nothing new
    if len(keep) >= K:
        break
    if t not in seen:
        keep.append(t); seen.add(t)

print(json.dumps({"keep": keep[:K]}))
