# TIER: strong
"""Insight: the rendezvous requirement makes TEAM FORMATION the real decision, not
per-ship tug picking. A team of k tugs that just finished a job together is already
SYNCED (identical free time, identical free position) -- routing it straight into the
next same-size job costs zero extra rendezvous wait, which beats reassembling a fresh
team from scratch even if the fresh candidates look individually closer.

For every ship (processed in release-time order, like the naive dispatcher) we compare
two options and take whichever gets there earliest and still fits a tide window:
  (a) REUSE the most recently synced team of the same crew size k, wherever it is;
  (b) build a FRESH team, but -- unlike the naive dispatcher -- prefer tugs that are
      already idle (free before this ship's release) over tugs that only look close
      because their last commitment happens to end nearby yet won't free up for a
      long time. That is exactly the straggler trap the naive nearest-by-raw-distance
      dispatcher falls into.
Whichever option wins gets committed and becomes the new "most recently synced team"
for that crew size, so a chain of same-size jobs threads a single persistent crew
through them, while unrelated jobs are still free to use other tugs in parallel."""
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


def try_window(job, s_cand):
    dur = job["dur"]
    for (o, c) in job["windows"]:
        s_try = max(s_cand, o)
        if s_try + dur <= c:
            return s_try
    return None


def main():
    T, N, L, pos, coeff, pen, jobs = read_input()
    free_time = [0] * T
    free_pos = list(pos)
    itin_last = [(0, pos[t]) for t in range(T)]
    itin_pts = [[(0, pos[t])] for t in range(T)]
    claims = []
    last_team = {}  # crew size k -> most recently synced team (list of tug ids)

    def commit(idx, s_final, team):
        job = jobs[idx]
        a, b, dur = job["a"], job["b"], job["dur"]
        for t in team:
            lt, lp = itin_last[t]
            if s_final > lt:
                itin_pts[t].append((s_final, a))
            itin_pts[t].append((s_final + dur, b))
            itin_last[t] = (s_final + dur, b)
            free_pos[t] = b
            free_time[t] = s_final + dur
        claims.append((idx + 1, s_final, list(team)))

    order = sorted(range(N), key=lambda idx: (jobs[idx]["release"], idx))
    for idx in order:
        job = jobs[idx]
        k, a, rel = job["k"], job["a"], job["release"]

        options = []  # (s_final, priority, team)  priority: reuse beats fresh on ties

        if k in last_team:
            team = last_team[k]
            arrival = max(free_time[t] + abs(free_pos[t] - a) for t in team)
            s_final = try_window(job, max(rel, arrival))
            if s_final is not None:
                options.append((s_final, 0, team))

        idle = [t for t in range(T) if free_time[t] <= rel]
        pool = idle if len(idle) >= k else list(range(T))
        pool_sorted = sorted(
            pool,
            key=lambda t: (0 if free_time[t] <= rel else 1, abs(free_pos[t] - a), free_time[t], t))
        fresh_team = pool_sorted[:k]
        arrival = max(free_time[t] + abs(free_pos[t] - a) for t in fresh_team)
        s_final = try_window(job, max(rel, arrival))
        if s_final is not None:
            options.append((s_final, 1, fresh_team))

        if not options:
            continue
        options.sort(key=lambda o: (o[0], o[1]))
        s_final, _, team = options[0]
        commit(idx, s_final, team)
        last_team[k] = team

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
