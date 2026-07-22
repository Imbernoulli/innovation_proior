# TIER: trivial
# Reproduces the checker's own internal baseline exactly: charge every bus the instant
# it arrives at full power, first-come-first-served over the shared berths.
import sys


def fcfs_schedule(buses, D, C, power_level):
    n = len(buses)
    occ = [0] * (D + 1)
    starts = [None] * n
    order = sorted(range(n), key=lambda i: (buses[i][0], i))
    for i in order:
        a_i, e_i = buses[i]
        L = -(-e_i // power_level) if e_i > 0 else 0
        if L == 0:
            starts[i] = a_i
            continue
        s = a_i
        placed = False
        while s + L <= D:
            if all(occ[t] < C for t in range(s, s + L)):
                placed = True
                break
            s += 1
        if not placed:
            s = max(a_i, D - L)
            placed = True
        starts[i] = s
        for t in range(s, min(D, s + L)):
            occ[t] += 1
    return starts


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); D = int(next(it)); C = int(next(it)); pmax = int(next(it))
    buses = []
    for _ in range(n):
        a_i = int(next(it)); theta0 = int(next(it)); e_i = int(next(it)); c10 = int(next(it))
        buses.append((a_i, theta0, e_i, c10))

    pairs = [(b[0], b[2]) for b in buses]
    starts = fcfs_schedule(pairs, D, C, pmax)

    out_lines = []
    for i, (a_i, theta0, e_i, c10) in enumerate(buses):
        power = [0] * D
        t = starts[i]
        remaining = e_i
        while remaining > 0 and t < D:
            p = min(pmax, remaining)
            power[t] = p
            remaining -= p
            t += 1
        out_lines.append(" ".join(map(str, power)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
