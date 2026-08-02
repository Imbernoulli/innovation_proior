# TIER: greedy
import sys, math

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    V = int(next(it)); T = int(next(it))
    variants = []
    for _ in range(V):
        fleet = int(next(it))
        p = float(next(it))
        reward = float(next(it))
        penalty = float(next(it))
        D = int(next(it))
        W = float(next(it))
        variants.append((fleet, p, reward, penalty, D, W))

    out = []
    for (fleet, p, reward, penalty, D, W) in variants:
        # The "obvious" canary-rollout instinct: start small and DOUBLE the cohort
        # every stage as long as previous stages "succeeded" -- paced only against
        # the previous stage's own size, never against the signal-latency window D.
        singleshot = math.floor(W / p) if p > 0 else 1
        c0 = max(1, singleshot // 3)
        stages = []
        t = 0
        c = c0
        used = 0
        while used < fleet and t < T:
            cc = min(c, fleet - used)
            stages.append((t, cc))
            used += cc
            t += 1
            c *= 2
        out.append(str(len(stages)))
        for (tt, cc) in stages:
            out.append("%d %d" % (tt, cc))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
