import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    t = [int(data[idx + i]) for i in range(n)]
    idx += n

    if n == 0:
        print(0)
        return

    # Brute force: try every assignment of each job to one of m presses
    # (with a load-symmetry break so equal-load presses aren't tried twice),
    # and take the minimum achievable makespan = max press load.
    best = [float('inf')]
    load = [0] * m

    def rec(i):
        if i == n:
            best[0] = min(best[0], max(load))
            return
        seen = set()
        for p in range(m):
            if load[p] in seen:
                continue
            seen.add(load[p])
            load[p] += t[i]
            rec(i + 1)
            load[p] -= t[i]

    rec(0)
    print(best[0])

main()
