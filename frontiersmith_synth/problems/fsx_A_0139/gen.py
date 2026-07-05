import sys

# Salmon migration ladder: low-discrepancy sampling COMPLETION.
# The monitoring grid is the unit square [0,1]^2:
#   x = fraction of the fish ladder climbed (0 = river mouth, 1 = spawning ground)
#   y = fraction of the daily cycle           (0 = midnight,   1 = next midnight)
# A total of M underwater cameras must sample this (position x time) plane as
# uniformly as possible (minimize star discrepancy). k cameras are ALREADY bolted
# in at legacy locations (given, fixed); the solver places the remaining M-k.
#
# Difficulty ladder: total camera count M grows small -> large. Everything below
# is a pure deterministic function of testId (LCG seeded by testId only).

LADDER_M = [24, 32, 40, 48, 56, 64, 72, 80, 84, 88]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER_M)) - 1
M = LADDER_M[idx]
k = max(3, round(0.25 * M))     # ~25% of cameras are pre-installed

# deterministic LCG seeded by testId only
seed = (1234567 + i * 7919) & 0x7fffffff
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff

# k legacy cameras, scattered over [0,1]^2 (never on a perfect lattice, so the
# completion genuinely has to compensate for their irregular placement)
fixed = []
for _ in range(k):
    x = rnd()
    y = rnd()
    fixed.append((x, y))

out = ["%d %d" % (M, k)]
for (x, y) in fixed:
    out.append("%.10f %.10f" % (x, y))
sys.stdout.write("\n".join(out) + "\n")
