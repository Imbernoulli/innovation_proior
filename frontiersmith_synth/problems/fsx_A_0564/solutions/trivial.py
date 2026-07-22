# TIER: trivial
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

    # canonical: each slot -> lowest-index qualified staff free that day, under maxshift
    used = [0] * S
    out = []
    for d in range(D):
        busy = set()
        for (k, h) in days[d]:
            pick = 0
            for j in range(S):
                if k in skills[j] and j not in busy and used[j] < maxshift[j]:
                    pick = j; break
            busy.add(pick); used[pick] += 1
            out.append(str(pick))
    sys.stdout.write(" ".join(out) + "\n")

main()
