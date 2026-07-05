# TIER: strong
# Greedy construction, then deterministic relocation local search under a fixed
# op/pass budget: repeatedly take each emitter out and drop it back on the plot
# that adds the most currently-uncovered deficit given all the OTHER emitters.
# Because removal is exact (coverage counts, not booleans), each relocation is a
# best-response move that never lowers the recovered union, so the layout climbs
# past plain greedy toward a locally optimal spread.  The overlap-inflated UB
# keeps the normalized score below 1.0.
import sys, json

inst = json.load(sys.stdin)
N, R, K = inst["N"], inst["R"], inst["K"]
grid = inst["grid"]

cnt = [[0] * N for _ in range(N)]   # how many emitters currently wet each plot


def bounds(r, c):
    return (max(0, r - R), max(0, c - R), min(N, r + R + 1), min(N, c + R + 1))


def marginal(r, c):
    r0, c0, r1, c1 = bounds(r, c)
    g = 0
    for rr in range(r0, r1):
        row = grid[rr]; cr = cnt[rr]
        for cc in range(c0, c1):
            if cr[cc] == 0:
                g += row[cc]
    return g


def add(r, c):
    r0, c0, r1, c1 = bounds(r, c)
    for rr in range(r0, r1):
        cr = cnt[rr]
        for cc in range(c0, c1):
            cr[cc] += 1


def remove(r, c):
    r0, c0, r1, c1 = bounds(r, c)
    for rr in range(r0, r1):
        cr = cnt[rr]
        for cc in range(c0, c1):
            cr[cc] -= 1


def best_position():
    best = (0, 0); best_g = -1
    for r in range(N):
        for c in range(N):
            g = marginal(r, c)
            if g > best_g:
                best_g = g; best = (r, c)
    return best, best_g


# --- greedy construction ---
emitters = []
for _ in range(K):
    (r, c), g = best_position()
    if g <= 0:
        break
    emitters.append([r, c]); add(r, c)

# --- relocation local search (bounded number of passes) ---
MAX_PASSES = 25
for _ in range(MAX_PASSES):
    improved = False
    for i in range(len(emitters)):
        r, c = emitters[i]
        remove(r, c)
        cur = marginal(r, c)              # what this emitter contributes right now
        (nr, nc), g = best_position()     # best spot given the others
        if g > cur:
            emitters[i] = [nr, nc]; add(nr, nc); improved = True
        else:
            add(r, c)                     # put it back where it was
    if not improved:
        break

print(json.dumps({"emitters": emitters}))
