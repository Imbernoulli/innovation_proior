# TIER: trivial
"""Fixed round-robin team assignment: no geometry, no idle-awareness, no chaining insight.
Processes ships in LISTED order and hands each one the next block of tug ids in a cyclic
rotation, regardless of where those tugs currently are or when they become free. This is
exactly the internal reference construction the checker itself builds (so this solution's
cost should land close to the checker's own baseline B, i.e. Ratio ~= 0.1)."""
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
    cursor = 0
    itin_last = [(0, pos[t]) for t in range(T)]  # last waypoint emitted per tug
    itin_pts = [[(0, pos[t])] for t in range(T)]
    claims = []

    for j in range(1, N + 1):
        job = jobs[j - 1]
        k = job["k"]
        chosen = [(cursor + i) % T for i in range(k)]
        cursor = (cursor + k) % T
        a, b, dur, rel = job["a"], job["b"], job["dur"], job["release"]
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
            # else: s_final == lt and lp == a already (guaranteed by construction)
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
