# TIER: strong
# Three-stage insight, composing all three mechanisms instead of one fixed local scan:
#
# 1) motif-length-diagnosis: find every LEFT-MAXIMAL matching pair of positions (i, j) --
#    i.e. seq[i-1] != seq[j-1] (or i == 0) so this pair isn't just a shifted echo of an
#    earlier pair -- and record the length of its maximal common run. Histogramming these
#    maximal lengths (not raw same-length-substring counts, which double count every
#    length nested inside a longer repeat) cleanly isolates the TRUE recurring scales: a
#    length-L motif that repeats produces one histogram hit at exactly L, not smeared
#    across every L' < L. Unrelated background coincidences are short and rare with this
#    alphabet, so the dominant, well-separated peaks are exactly the planted scales, even
#    when two different-length families are interleaved.
# 2) greedy-dictionary-build: for each diagnosed scale, count exact recurring L-mers and
#    keep only the ones whose expected net savings -- (extra reuses) * (literal cost minus
#    pointer cost) minus the one-time dictionary header -- is positive. This is what makes
#    the dictionary choice a real decision: a motif recurring once isn't worth declaring.
# 3) dp-segment-boundary: given that fixed dictionary, a forward shortest-path DP over
#    positions finds the CHEAPEST way to cut the sequence into literal/reference segments
#    -- globally optimal for the fixed dictionary, unlike a left-to-right greedy longest-
#    match scan, which can lock in a locally-longest match that blocks a cheaper overall
#    parse. Entries the DP ends up never using are dropped before scoring, so wasted
#    dictionary candidates cost nothing.
import sys, json

inst = json.load(sys.stdin)
seq = inst["seq"]
n = inst["n"]
BITS_PER_SYMBOL = inst["bits_per_symbol"]
PTR_BITS = inst["ptr_bits"]
DICT_HEADER_BITS = inst["dict_header_bits"]
MAX_MOTIF_LEN = inst["max_motif_len"]
MAX_DICT_ENTRIES = inst["max_dict_entries"]


def dict_entry_cost(L):
    return DICT_HEADER_BITS + BITS_PER_SYMBOL * L


MINLEN = 3
MAXPROBE = min(MAX_MOTIF_LEN, max(MINLEN, n // 3))

# ---- 1) motif-length diagnosis via left-maximal common-run histogram ----
hist = {}
for i in range(n - MINLEN):
    si = seq[i]
    for j in range(i + MINLEN, n):
        if seq[j] != si:
            continue
        if i > 0 and seq[i - 1] == seq[j - 1]:
            continue  # shifted echo of an earlier (i-1, j-1) pair -- skip
        L = 1
        max_l = min(MAXPROBE, n - j)
        while L < max_l and seq[i + L] == seq[j + L]:
            L += 1
        if L >= MINLEN:
            hist[L] = hist.get(L, 0) + 1

order = sorted(hist.keys(), key=lambda L: -hist[L] * L)
picked = []
for L in order:
    if hist[L] < 2:
        continue
    if all(abs(L - p) > 1 for p in picked):
        picked.append(L)
    if len(picked) >= 3:
        break

# ---- 2) greedy dictionary build: keep candidates with positive net savings ----
candidates = []
for L in picked:
    cnt = {}
    for i in range(0, n - L + 1):
        key = tuple(seq[i:i + L])
        cnt[key] = cnt.get(key, 0) + 1
    for content, c in cnt.items():
        if c < 2:
            continue
        net = (c - 1) * (BITS_PER_SYMBOL * L - PTR_BITS) - dict_entry_cost(L)
        if net > 0:
            candidates.append((net, content))
candidates.sort(key=lambda t: -t[0])

dictionary = []
seen = set()
for net, content in candidates:
    if content in seen:
        continue
    seen.add(content)
    dictionary.append(content)
    if len(dictionary) >= MAX_DICT_ENTRIES:
        break

# ---- 3) dp-segment-boundary over the fixed candidate dictionary, then prune unused ----
INF = float("inf")
dp = [INF] * (n + 1)
par = [(-1, -1)] * (n + 1)  # (prev_pos, kind); kind=-1 literal step, else dict idx
dp[0] = 0

matches_at = [[] for _ in range(n + 1)]
for idx, content in enumerate(dictionary):
    L = len(content)
    if L == 0 or L > n:
        continue
    for i in range(0, n - L + 1):
        if tuple(seq[i:i + L]) == content:
            matches_at[i].append((L, idx))

for i in range(n):
    if dp[i] == INF:
        continue
    if dp[i] + BITS_PER_SYMBOL < dp[i + 1]:
        dp[i + 1] = dp[i] + BITS_PER_SYMBOL
        par[i + 1] = (i, -1)
    for L, idx in matches_at[i]:
        if dp[i] + PTR_BITS < dp[i + L]:
            dp[i + L] = dp[i] + PTR_BITS
            par[i + L] = (i, idx)

used = set()
pos = n
raw_segs = []
while pos > 0:
    p, kind = par[pos]
    if kind == -1:
        raw_segs.append(("lit", p, pos))
    else:
        used.add(kind)
        raw_segs.append(("ref", p, pos, kind))
    pos = p
raw_segs.reverse()

remap = {old: new for new, old in enumerate(sorted(used))}
out_dict = [list(dictionary[old]) for old in sorted(used)]
out_segs = []
for s in raw_segs:
    if s[0] == "lit":
        out_segs.append({"type": "lit", "len": s[2] - s[1]})
    else:
        out_segs.append({"type": "ref", "dict_idx": remap[s[3]]})

print(json.dumps({"dictionary": out_dict, "segments": out_segs}))
