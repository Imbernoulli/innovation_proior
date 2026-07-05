# TIER: strong
# Greedy construction + relocation local search with seeded perturbation restarts.
#
# Start from the marginal-cost greedy plan, then repeatedly relocate single cars
# to the track that most reduces the official (affinity-weighted) cost, using an
# O(B) incremental delta.  At a local optimum, apply a deterministic random kick
# (reassign a handful of cars) and re-descend, always keeping the best plan seen.
# All randomness is seeded from the instance, so the run is reproducible.  Because
# more blocks than tracks force mixing the lower bound ignores, the normalized
# score stays below 1.0 -> genuine headroom.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_cars"]
K = inst["n_tracks"]
C = inst["cap"]
B = inst["n_blocks"]
block = inst["block"]
W = inst["mix_w"]
split_pen = inst["split_pen"]
over_pen = inst["over_pen"]


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


rnd = _rng((N * 131 + K * 17 + C * 7 + 1) & 0xFFFFFFFF)


def build_state(assign):
    counts = [0] * K
    cnt = [[0] * B for _ in range(K)]
    for i in range(N):
        t = assign[i]
        counts[t] += 1
        cnt[t][block[i]] += 1
    return counts, cnt


def track_mix(row, b):
    Wb = W[b]
    m = 0
    for b2 in range(B):
        c = row[b2]
        if c:
            m += c * Wb[b2]
    return m


def move_delta(counts, cnt, i, s, d):
    if s == d:
        return 0
    b = block[i]
    # mix: remove from s (W[b][b]=0 so own-block term is 0), add to d
    mix_delta = -track_mix(cnt[s], b) + track_mix(cnt[d], b)
    # split: change in sum_k C(cnt[k][b],2) is -(cnt[s][b]-1) + cnt[d][b]
    split_delta = split_pen * ((cnt[s][b] - 1) - cnt[d][b])
    ov = (max(0, counts[s] - 1 - C) - max(0, counts[s] - C)
          + max(0, counts[d] + 1 - C) - max(0, counts[d] - C))
    return over_pen * ov + mix_delta + split_delta


def apply_move(assign, counts, cnt, i, d):
    s = assign[i]
    b = block[i]
    counts[s] -= 1
    cnt[s][b] -= 1
    counts[d] += 1
    cnt[d][b] += 1
    assign[i] = d


def cost_of(assign):
    counts, cnt = build_state(assign)
    overflow = sum(max(0, c - C) for c in counts)
    mix_cost = 0
    for k in range(K):
        row = cnt[k]
        for b in range(B):
            cb = row[b]
            if cb:
                Wb = W[b]
                for b2 in range(b + 1, B):
                    cb2 = row[b2]
                    if cb2:
                        mix_cost += cb * cb2 * Wb[b2]
    Q = 0
    for k in range(K):
        row = cnt[k]
        for b in range(B):
            v = row[b]
            if v > 1:
                Q += v * (v - 1) // 2
    Nb = [0] * B
    for b in block:
        Nb[b] += 1
    sum_block = sum(x * (x - 1) // 2 for x in Nb)
    return over_pen * overflow + mix_cost + split_pen * (sum_block - Q)


def greedy_init():
    counts = [0] * K
    cnt = [[0] * B for _ in range(K)]
    assign = [0] * N
    for i in range(N):
        b = block[i]
        best_t, best_delta = 0, None
        for t in range(K):
            delta = track_mix(cnt[t], b) - split_pen * cnt[t][b]
            if counts[t] >= C:
                delta += over_pen
            if best_delta is None or delta < best_delta:
                best_delta, best_t = delta, t
        assign[i] = best_t
        counts[best_t] += 1
        cnt[best_t][b] += 1
    return assign


def descend(assign):
    counts, cnt = build_state(assign)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            s = assign[i]
            best_d, best_delta = s, 0
            for d in range(K):
                if d == s:
                    continue
                delta = move_delta(counts, cnt, i, s, d)
                if delta < best_delta:
                    best_delta, best_d = delta, d
            if best_d != s:
                apply_move(assign, counts, cnt, i, best_d)
                improved = True
    return assign


cur = descend(greedy_init())
best = list(cur)
best_cost = cost_of(best)

RESTARTS = 16
for _ in range(RESTARTS):
    trial = list(best)
    kicks = rnd(2, max(2, N // 10))
    for _ in range(kicks):
        i = rnd(0, N - 1)
        trial[i] = rnd(0, K - 1)
    trial = descend(trial)
    c = cost_of(trial)
    if c < best_cost:
        best_cost = c
        best = list(trial)

print(json.dumps({"assign": best}))
