# TIER: trivial
# Constrained NEXT-FIT: fill the current loop until a well doesn't fit -- on flow
# OR temperature band -- then open a fresh loop and never look back.  This
# reproduces the evaluator's weak reference operator, so it scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
flow = inst["flow"]
temp = inst["temp"]
C = inst["capacity"]
band = inst["band"]
N = inst["n"]

assign = [0] * N
cur = -1          # index of current open loop; -1 = none yet
rem = 0
tmin = tmax = 0
next_idx = 0
for i in range(N):
    f = flow[i]
    t = temp[i]
    placed = False
    if cur >= 0 and f <= rem:
        nmin = t if t < tmin else tmin
        nmax = t if t > tmax else tmax
        if nmax - nmin <= band:
            assign[i] = cur
            rem -= f
            tmin, tmax = nmin, nmax
            placed = True
    if not placed:
        cur = next_idx
        next_idx += 1
        assign[i] = cur
        rem = C - f
        tmin = tmax = t

print(json.dumps({"assign": assign}))
