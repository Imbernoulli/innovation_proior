# TIER: greedy
import sys

# The obvious first idea: "race to idle". Run at the top DVFS level
# whenever any job's window is open, drop straight to idle the instant
# nothing is pending. This looks efficient (never runs when there's
# nothing to do) but (a) always pays the full cubic-top-speed power
# while busy even when a much cheaper sustained rate would clear the
# same work, and (b) always switches level exactly AT the moment work
# resumes, so the ramp-loss from that switch lands on the first slot
# the job actually needs -- eating into exactly the capacity the
# deadline was counting on.


def read_instance():
    toks = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = toks[pos[0]]
        pos[0] += 1
        return v

    T = int(nxt()); m = int(nxt()); J = int(nxt())
    for _ in range(m):
        nxt(); nxt()
    nxt()  # ramp
    for _ in range(m):
        for _ in range(m):
            nxt()
    jobs = []
    for _ in range(J):
        r = int(nxt()); d = int(nxt()); w = int(nxt())
        jobs.append((r, d, w))
    return T, m, jobs


def main():
    T, m, jobs = read_instance()
    busy = [False] * T
    for (r, d, w) in jobs:
        for t in range(r, min(d, T)):
            busy[t] = True
    levels = [(m - 1) if busy[t] else 0 for t in range(T)]
    print(" ".join(str(x) for x in levels))


if __name__ == "__main__":
    main()
