# TIER: greedy
# Marginal-gain greedy maximum coverage: place emitters one at a time, each time
# choosing the plot whose block adds the MOST currently-uncovered deficit.  This
# naturally spreads emitters across separate hot zones (once a zone is wet, a
# second emitter there adds little), beating the top-block pile-up.  Still myopic
# and never revisits earlier choices, so it stays below the local-search tier.
import sys, json

inst = json.load(sys.stdin)
N, R, K = inst["N"], inst["R"], inst["K"]
grid = inst["grid"]

covered = [[False] * N for _ in range(N)]


def marginal(r, c):
    r0 = max(0, r - R); c0 = max(0, c - R)
    r1 = min(N, r + R + 1); c1 = min(N, c + R + 1)
    g = 0
    for rr in range(r0, r1):
        row = grid[rr]; cov = covered[rr]
        for cc in range(c0, c1):
            if not cov[cc]:
                g += row[cc]
    return g


def apply(r, c):
    r0 = max(0, r - R); c0 = max(0, c - R)
    r1 = min(N, r + R + 1); c1 = min(N, c + R + 1)
    for rr in range(r0, r1):
        cov = covered[rr]
        for cc in range(c0, c1):
            cov[cc] = True


emitters = []
for _ in range(K):
    best = None; best_g = -1
    for r in range(N):
        for c in range(N):
            g = marginal(r, c)
            if g > best_g:
                best_g = g; best = (r, c)
    if best is None or best_g <= 0:
        break
    emitters.append([best[0], best[1]])
    apply(best[0], best[1])

print(json.dumps({"emitters": emitters}))
