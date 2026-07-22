# TIER: strong
# Insight: catalyst freshness is an inventory to be SPENT at the price peaks, not a
# maintenance level to be defended.  What matters is the TIMING of throughput, not its
# amount.  So: idle through the cheap low-demand valleys to conserve freshness, park the
# offline regenerations inside those valleys so each peak is met with a fresh reactor,
# and run flat-out only during the high-price demand windows.
import sys

lines = sys.stdin.read().split("\n")
T, R, Q, d, L = [int(x) for x in lines[0].split()]
e = [float(x) for x in lines[1].split()]
p = [float(x) for x in lines[2].split()]
cap = [float(x) for x in lines[3].split()]

valley_p = min(p)
ispeak = [p[t] > valley_p + 1e-9 for t in range(T)]

# collect peak window start indices
peak_starts = []
t = 0
while t < T:
    if ispeak[t] and (t == 0 or not ispeak[t - 1]):
        peak_starts.append(t)
    t += 1


def build_row():
    sched = [0] * T          # idle valleys by default (conserve freshness)
    occ = [False] * T
    # reserve a fresh reactor for each peak: place a regeneration block of length d
    # ending immediately before the peak, entirely inside the preceding valley.
    for ps in peak_starts:
        s = ps - d
        if s >= 0 and all((not ispeak[k]) and (not occ[k]) for k in range(s, ps)):
            for k in range(s, ps):
                occ[k] = True
                sched[k] = -1
    # run flat-out during peak windows
    for t in range(T):
        if ispeak[t]:
            sched[t] = Q
    return sched


row = build_row()
line = " ".join(str(v) for v in row)
sys.stdout.write("\n".join(line for _ in range(R)) + "\n")
