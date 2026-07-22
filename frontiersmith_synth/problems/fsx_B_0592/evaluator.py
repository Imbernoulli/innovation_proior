import sys, json, random, isorun
from collections import Counter

# ==========================================================================
# fsx_B_0592 -- drift-aware-heavy-hitter-budget (Format B, isolated candidate)
# Theme: a fixed-memory streaming summarizer that must retain the EVENTUAL
# heavy hitters despite a mid-stream distribution drift.
#
# The PUBLIC instance is an OBSERVED PREFIX of a token stream plus a memory
# budget K.  The candidate must commit a bounded summary: a set of <= K token
# ids it "keeps in memory" at the window close.  The stream then CONTINUES in a
# HIDDEN suffix (never shown to the candidate); the true global top-M heavy
# hitters are computed over prefix+suffix.  A heavy hitter can only be recovered
# if it was kept.  Score = recall of the true top-M inside the kept set.
#
# Planted structure (the trap):
#   * PERSIST tokens are heavy throughout (a small stable floor).
#   * LEADERS dominate the pre-drift region then vanish -- huge prefix counts,
#     zero global relevance.
#   * DECOY risers spike EARLY in the post-drift region (high prefix count,
#     front-loaded) then fizzle in the suffix -- they bait a frequency-greedy.
#   * TRUE risers ramp LATE (back-loaded, low prefix count) and SURGE in the
#     hidden suffix -- they are the eventual heavy hitters.
#   A frequency-greedy (keep the most-counted-so-far) spends its whole budget on
#   LEADERS/DECOYS and recovers almost none of the true risers.
#   The insight: detect the drift, exponentially deweight pre-drift counts, and
#   reserve tail slots so the late, accelerating risers out-rank the stale
#   leaders and survive in memory.
# Objective: MAXIMIZE recall.  Deterministic; the suffix is hidden.
# ==========================================================================

P_IDS = [0, 1]                      # PERSIST (stable heavy hitters)
LN = 14                             # LEADERS (pre-drift dominators)
RD = 14                            # DECOY risers (early spike, then fizzle)
RT = 22                            # TRUE risers (late ramp, suffix surge)
NN = 160                           # NOISE tokens
K = 16                             # memory budget
M = 12                             # number of heavy hitters to recover

LEAD_IDS = list(range(100, 100 + LN))
RISE_IDS = list(range(200, 200 + RT))
DECOY_IDS = list(range(300, 300 + RD))
NOISE_IDS = list(range(1000, 1000 + NN))


def build_instance(seed):
    rng = random.Random(seed)
    N1 = 900 + rng.randint(0, 300)          # pre-drift length
    R2 = 1800 + rng.randint(0, 600)         # post-drift prefix length
    Npre = N1 + R2

    occ = []                                # (position, token) prefix occurrences
    suf = {}                                # token -> hidden suffix count

    def add(tok, count, lo, hi):
        for _ in range(count):
            occ.append((rng.uniform(lo, hi), tok))

    # PERSIST: heavy in every region.
    for t in P_IDS:
        add(t, 60 + rng.randint(0, 20), 0, N1)
        add(t, 88 + rng.randint(0, 24), N1, Npre)
        suf[t] = 128 + rng.randint(0, 26)

    # LEADERS: pre-drift only, then gone.
    for t in LEAD_IDS:
        add(t, 50 + rng.randint(0, 12), 0, N1)
        add(t, rng.randint(0, 4), N1, Npre)
        suf[t] = 0

    # DECOY risers: FRONT-loaded spike in region 2, fizzle in suffix.
    d_lo, d_hi = N1, N1 + int(0.48 * R2)
    for t in DECOY_IDS:
        add(t, 42 + rng.randint(0, 14), d_lo, d_hi)
        suf[t] = rng.randint(0, 3)

    # TRUE risers: BACK-loaded ramp in region 2, SURGE in the hidden suffix.
    r_lo, r_hi = N1 + int(0.45 * R2), Npre
    for t in RISE_IDS:
        q = rng.random()                            # latent quality
        ramp = 16 + int(round(15 * (0.7 * q + 0.3 * rng.random())))
        add(t, ramp, r_lo, r_hi)
        # suffix surge only WEAKLY correlated with the prefix ramp -> the exact
        # top-M identity is not perfectly predictable from the prefix.
        surge = 60 + int(round(210 * (0.4 * q + 0.6 * rng.random())))
        suf[t] = surge

    # NOISE: low everywhere.
    for t in NOISE_IDS:
        add(t, rng.randint(1, 3), 0, Npre)
        suf[t] = rng.randint(0, 2)

    occ.sort(key=lambda z: (z[0], z[1]))
    stream = [t for _, t in occ]

    pc = Counter(stream)
    gc = {}
    for t in set(pc) | set(suf):
        gc[t] = pc.get(t, 0) + suf.get(t, 0)
    ranked = sorted(gc, key=lambda t: (-gc[t], t))
    top_m = ranked[:M]

    public = {"K": K, "M": M, "stream": stream}
    hidden = {"top_m": top_m}
    return {"public": public, "hidden": hidden}


def make_instances():
    return [build_instance(4200 + 7 * s) for s in range(10)]


def score(inst, ans):
    pub = inst["public"]
    kk = pub["K"]
    seen = set(pub["stream"])
    top_m = inst["hidden"]["top_m"]
    if not isinstance(ans, dict) or "keep" not in ans:
        return False, 0.0
    keep = ans["keep"]
    if not isinstance(keep, list) or len(keep) > kk:
        return False, 0.0
    ks = set()
    for v in keep:
        if isinstance(v, bool) or not isinstance(v, int):
            return False, 0.0
        if v not in seen:                 # bounded memory cannot hold an unseen token
            return False, 0.0
        ks.add(v)
    if len(ks) != len(keep):              # no duplicates
        return False, 0.0
    hit = sum(1 for t in top_m if t in ks)
    return True, hit / float(len(top_m))


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, r = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
