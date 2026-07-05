# TIER: greedy
"""Spread the population as flatly as possible across all pools
(water-filling under the capacities). A flat profile gives c1 ~= 2,
much better than piling it in one place."""
import sys


def waterfill(cap, S):
    n = len(cap)
    f = [0.0] * n
    active = [True] * n
    r = float(S)
    for _ in range(n + 5):
        na = sum(1 for a in active if a)
        if r <= 1e-12 or na == 0:
            break
        lvl = r / na
        hit = [i for i in range(n) if active[i] and (cap[i] - f[i]) < lvl]
        if hit:
            for i in hit:
                r -= (cap[i] - f[i])
                f[i] = float(cap[i])
                active[i] = False
        else:
            for i in range(n):
                if active[i]:
                    f[i] += lvl
            r = 0.0
    return f


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); S = int(tok[1])
    cap = [int(x) for x in tok[2:2 + n]]
    f = waterfill(cap, S)
    # fix any residual so the sum is exactly S
    diff = S - sum(f)
    for i in range(n):
        if diff == 0:
            break
        room = cap[i] - f[i] if diff > 0 else f[i]
        step = diff if abs(diff) <= room else (room if diff > 0 else -room)
        f[i] += step
        diff -= step
    sys.stdout.write(" ".join("%.12g" % x for x in f) + "\n")


if __name__ == "__main__":
    main()
