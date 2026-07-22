# TIER: trivial
# Reproduces the checker's fully-serialized baseline: dispatch in release order,
# each train departs only after the previous one has cleared the whole line
# (trains run straight through, no meets, no dwell). -> ~0.1
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    S = int(next(it)); TMAX = int(next(it))
    cap = [int(next(it)) for _ in range(S)]
    N = int(next(it))
    trains = []
    for _ in range(N):
        di = int(next(it)); r = int(next(it)); dd = int(next(it))
        w = int(next(it)); v = int(next(it)); h = int(next(it))
        trains.append((di, r, dd, w, v, h))

    order = sorted(range(N), key=lambda j: (trains[j][1], j))
    dep = [0] * N
    clear = 0
    for j in order:
        di, r, dd, w, v, h = trains[j]
        dep[j] = max(r, clear)
        clear = dep[j] + (S - 1) * h

    out = []
    for j in range(N):
        di, r, dd, w, v, h = trains[j]
        e = [dep[j] + k * h for k in range(S - 1)]
        out.append(" ".join(str(x) for x in e))
    sys.stdout.write("\n".join(out) + "\n")


main()
