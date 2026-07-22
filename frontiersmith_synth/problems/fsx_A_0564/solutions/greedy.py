# TIER: greedy
# Obvious approach: balance the workload evenly (assign each slot to the qualified
# staff with the fewest shifts so far). Ignores the scenarios and the repair rule.
import sys

def main():
    toks = sys.stdin.read().split()
    it = iter(toks); nxt = lambda: next(it)
    S = int(nxt()); D = int(nxt()); C = int(nxt())
    maxshift = [int(nxt()) for _ in range(S)]
    base = [int(nxt()) for _ in range(S)]
    skills = []
    for _ in range(S):
        cnt = int(nxt()); skills.append(set(int(nxt()) for _ in range(cnt)))
    days = []
    for _ in range(D):
        T = int(nxt()); days.append([(int(nxt()), int(nxt())) for _ in range(T)])

    load = [0] * S
    out = []
    for d in range(D):
        busy = set()
        for (k, h) in days[d]:
            best = -1; bl = None
            for j in range(S):
                if k in skills[j] and j not in busy and load[j] < maxshift[j]:
                    if bl is None or load[j] < bl:
                        bl = load[j]; best = j
            if best < 0:
                best = 0
            busy.add(best); load[best] += 1
            out.append(str(best))
    sys.stdout.write(" ".join(out) + "\n")

main()
