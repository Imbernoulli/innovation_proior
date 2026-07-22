# TIER: greedy
# The obvious maintenance recipe: run every reactor flat-out, and regenerate the moment
# the marginal conversion drops below a fixed threshold.  Ignores prices and demand
# entirely -> spends fresh catalyst on cheap valleys and meets the price peaks stale.
import sys

lines = sys.stdin.read().split("\n")
T, R, Q, d, L = [int(x) for x in lines[0].split()]
e = [float(x) for x in lines[1].split()]

THRESH = 0.45  # regenerate when the next feed unit converts below this rate


def marginal(w):
    return e[w] if w < L else e[L - 1]


def plan():
    sched = []
    w = 0
    t = 0
    while t < T:
        if marginal(w) < THRESH and t + d <= T:
            sched.extend([-1] * d)
            w = 0
            t += d
        else:
            sched.append(Q)
            w += Q
            t += 1
    return sched[:T]


row = plan()
line = " ".join(str(v) for v in row)
sys.stdout.write("\n".join(line for _ in range(R)) + "\n")
