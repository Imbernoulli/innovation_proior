# TIER: greedy
# Marginal-cost greedy.  Process cars in the given order; place each car on the
# track whose incremental cost is smallest.  Adding a car of block b to track t:
#   mix delta   = sum_{b'} cnt_t[b'] * W[b][b']       # weighted affinity to cars on t
#   split delta = -split_pen * cnt_t[b]               # same-block pairs now joined
#   over delta  = over_pen if count_t >= cap else 0   # a new over-capacity car
# This clusters same-block cars and respects affinities, beating scattered
# round-robin, but its myopic arrival-order commitment leaves the weighted
# block-to-track partition far from optimal (no re-optimization).
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

counts = [0] * K
cnt = [[0] * B for _ in range(K)]
assign = [0] * N

for i in range(N):
    b = block[i]
    Wb = W[b]
    best_t = 0
    best_delta = None
    for t in range(K):
        row = cnt[t]
        mix = 0
        for b2 in range(B):
            c = row[b2]
            if c:
                mix += c * Wb[b2]
        delta = mix - split_pen * row[b]
        if counts[t] >= C:
            delta += over_pen
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_t = t
    assign[i] = best_t
    counts[best_t] += 1
    cnt[best_t][b] += 1

print(json.dumps({"assign": assign}))
