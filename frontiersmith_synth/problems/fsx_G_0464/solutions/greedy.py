# TIER: greedy
# A reasonable first fix: kill the two easy nuisances but skip the subtle one.
#   (1) L1-normalize each descriptor into a "composition" -> removes the per-protein
#       abundance / length factor (scale-invariant).
#   (2) divide each channel by its across-batch standard deviation -> equalizes the
#       loud nuisance channels against the quiet informative ones.
# It does NOT center out the per-channel baseline offsets, so the large constant
# offsets still tilt the cosine and it retrieves only a little better than raw
# distance -- above trivial, well below a properly standardized kernel.
import sys, json, math

inst = json.load(sys.stdin)
N = inst["n"]
D = inst["dim"]
feats = inst["features"]

comp = []
for v in feats:
    s = 0.0
    for x in v:
        s += abs(x)
    if s < 1e-12:
        s = 1e-12
    comp.append([x / s for x in v])

std = [0.0] * D
for d in range(D):
    ss = 0.0
    for i in range(N):
        ss += comp[i][d] * comp[i][d]
    sd = math.sqrt(ss / N)
    std[d] = sd if sd > 1e-12 else 1e-12

z = [[comp[i][d] / std[d] for d in range(D)] for i in range(N)]
zn = []
for v in z:
    nn = math.sqrt(sum(x * x for x in v))
    zn.append(nn if nn > 1e-12 else 1e-12)

ranking = []
for i in range(N):
    zi = z[i]
    ni = zn[i]
    scored = []
    for j in range(N):
        if j == i:
            continue
        dot = 0.0
        zj = z[j]
        for a, b in zip(zi, zj):
            dot += a * b
        scored.append((dot / (ni * zn[j]), j))
    scored.sort(key=lambda t: (-t[0], t[1]))
    ranking.append([j for _, j in scored])

print(json.dumps({"ranking": ranking}))
