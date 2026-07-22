# TIER: greedy
"""The obvious per-ship dispatcher: process ships in release-time order and, for each one,
grab the k tugs that are geographically NEAREST right now (breaking ties by whichever
becomes free soonest). This ignores whether a nearby tug is actually free soon -- a tug
that currently looks close (because its last committed job happens to end near here) but
won't actually be free for a long time gets picked anyway, and the whole team then waits on
that straggler. On corridors where a fast-turnaround persistent team is available a little
farther off, this costs missed tide windows that a team-aware planner would have avoided."""
import sys


def read_input():
    toks = sys.stdin.read().split()
    it = iter(toks)
    T = int(next(it)); N = int(next(it)); L = int(next(it))
    pos = [int(next(it)) for _ in range(T)]
    coeff = int(next(it)); pen = int(next(it))
    jobs = []
    for _ in range(N):
        a = int(next(it)); b = int(next(it)); k = int(next(it))
        rel = int(next(it)); w = int(next(it)); nw = int(next(it))
        windows = [(int(next(it)), int(next(it))) for _ in range(nw)]
        jobs.append({"a": a, "b": b, "k": k, "release": rel, "weight": w,
                     "windows": windows, "dur": abs(b - a)})
    return T, N, L, pos, coeff, pen, jobs


def main():
    T, N, L, pos, coeff, pen, jobs = read_input()
    free_time = [0] * T
    free_pos = list(pos)
    itin_last = [(0, pos[t]) for t in range(T)]
    itin_pts = [[(0, pos[t])] for t in range(T)]
    claims = []

    order = sorted(range(N), key=lambda idx: (jobs[idx]["release"], idx))
    for idx in order:
        j = idx + 1
        job = jobs[idx]
        k = job["k"]
        a, b, dur, rel = job["a"], job["b"], job["dur"], job["release"]
        cand = sorted(range(T), key=lambda t: (abs(free_pos[t] - a), free_time[t], t))
        chosen = cand[:k]
        arrival = max(free_time[t] + abs(free_pos[t] - a) for t in chosen)
        s_cand = max(rel, arrival)
        s_final = None
        for (o, c) in job["windows"]:
            s_try = max(s_cand, o)
            if s_try + dur <= c:
                s_final = s_try
                break
        if s_final is None:
            continue
        for t in chosen:
            lt, lp = itin_last[t]
            if s_final > lt:
                itin_pts[t].append((s_final, a))
            itin_pts[t].append((s_final + dur, b))
            itin_last[t] = (s_final + dur, b)
            free_pos[t] = b
            free_time[t] = s_final + dur
        claims.append((j, s_final, chosen))

    out = []
    for t in range(T):
        pts = itin_pts[t]
        out.append(str(len(pts)) + " " + " ".join(f"{tt} {xx}" for tt, xx in pts))
    out.append(str(len(claims)))
    for j, s_final, chosen in claims:
        out.append(f"{j} {s_final} " + " ".join(map(str, chosen)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
