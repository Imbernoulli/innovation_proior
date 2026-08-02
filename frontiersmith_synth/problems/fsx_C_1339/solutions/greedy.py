# TIER: greedy
# Textbook "split application" recipe: everyone knows dumping the whole season's
# fertilizer on day 1 is bad, so spread it into P EQUAL doses on P evenly spaced
# calendar days, applying all three nutrients together each visit (one trip to
# the field, fertilize everything). This is standard agronomic advice ("apply
# a third at planting, a third at 30 days, a third at 60 days") and avoids the
# up-front leaching trap, but it never reads the demand curve's actual shape
# (peak timing, width) and never separates K and Mg in time, so it walks
# straight into the ion-antagonism penalty whenever their curves overlap and
# wastes lumps on days with little real demand.
import sys

toks = sys.stdin.read().split()
p = iter(toks)
T = int(next(p)); P = int(next(p))
vN = float(next(p)); vK = float(next(p)); vMg = float(next(p))
rN = float(next(p)); rK = float(next(p)); rMg = float(next(p))
kappa = float(next(p))
BN = float(next(p)); BK = float(next(p)); BMg = float(next(p))
for _ in range(T):
    next(p); next(p); next(p)  # demand curves unused: equal-split ignores curve shape

k = max(1, min(P, T))
if k == 1:
    days = [0]
else:
    days = sorted(set(round(i * (T - 1) / (k - 1)) for i in range(k)))

AN = [0.0] * T; AK = [0.0] * T; AMg = [0.0] * T
shareN = BN / len(days)
shareK = BK / len(days)
shareMg = BMg / len(days)
for d in days:
    AN[d] += shareN
    AK[d] += shareK
    AMg[d] += shareMg

out = []
for t in range(T):
    out.append("%.6f %.6f %.6f" % (AN[t], AK[t], AMg[t]))
sys.stdout.write("\n".join(out) + "\n")
