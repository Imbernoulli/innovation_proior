# TIER: strong
"""The insight: contamination is a one-way ratchet, so a robot that is ALWAYS
sent to the same grade never needs to decon at all. Partition the fleet into
per-grade castes (sized by workload, via largest-remainder apportionment) so
the bulk of jobs are served by a robot that is already the right grade -- zero
decon cost. A small floater pool mops up jobs a caste cannot finish on time;
floaters process their queue in synchronized rounds so that whenever several
of them need a grade change at the same time, they are grouped into ONE
shared, capacity-C decon cycle instead of each firing its own -- treating the
rare cross-caste transition as a batched, scheduled commodity."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); R = int(next(it)); J = int(next(it))
    T = int(next(it)); C = int(next(it)); KCOST = int(next(it))

    jobs = []
    for _ in range(J):
        g = int(next(it)); rel = int(next(it)); dur = int(next(it))
        dl = int(next(it)); w = int(next(it))
        jobs.append((g, rel, dur, dl, w))

    jobs_by_grade = {g: [] for g in range(1, L + 1)}
    total_dur = [0] * (L + 1)
    for jid in range(1, J + 1):
        g = jobs[jid - 1][0]
        jobs_by_grade[g].append(jid)
        total_dur[g] += jobs[jid - 1][2]

    total = sum(total_dur[1:L + 1])
    S = max(1, round(R * 0.2))
    if R - S < L:
        S = max(0, R - L)
    Rc = R - S

    if total <= 0 or Rc <= 0:
        base = [Rc // L] * L
        rem = Rc - sum(base)
        for i in range(rem):
            base[i] += 1
        caste_size = base
    else:
        shares = [total_dur[g] * Rc / total for g in range(1, L + 1)]
        floor_shares = [int(x) for x in shares]
        rem = Rc - sum(floor_shares)
        order = sorted(range(L), key=lambda i: -(shares[i] - floor_shares[i]))
        caste_size = floor_shares[:]
        for i in range(max(0, rem)):
            caste_size[order[i % L]] += 1

    caste_robots = {}
    rid = 1
    for gi in range(L):
        g = gi + 1
        caste_robots[g] = list(range(rid, rid + caste_size[gi]))
        rid += caste_size[gi]
    floaters = list(range(rid, R + 1))
    if not floaters:
        biggest = max(range(1, L + 1), key=lambda g: len(caste_robots[g]))
        if caste_robots[biggest]:
            floaters = [caste_robots[biggest].pop()]
        else:
            floaters = [R]  # degenerate R==0 guard (never happens per constraints)

    robot_free = {r: 0 for r in range(1, R + 1)}
    robot_events = {r: [] for r in range(1, R + 1)}
    overflow = []

    for g in range(1, L + 1):
        crew = caste_robots[g]
        joblist = sorted(jobs_by_grade[g], key=lambda j: (jobs[j - 1][1], j))
        if not crew:
            overflow.extend(joblist)
            continue
        for jid in joblist:
            gg, rel, dur, dl, w = jobs[jid - 1]
            r = min(crew, key=lambda x: (robot_free[x], x))
            s = max(robot_free[r], rel)
            finish = s + dur
            if finish > dl:
                overflow.append(jid)
                continue
            robot_events[r].append(("J", jid, s))
            robot_free[r] = finish

    overflow.sort(key=lambda j: (jobs[j - 1][1], j))
    Sn = len(floaters)
    floater_jobs = {f: [] for f in floaters}
    for i, jid in enumerate(overflow):
        f = floaters[i % Sn]
        floater_jobs[f].append(jid)

    cur_time = {f: 0 for f in floaters}
    cur_contam = {f: L for f in floaters}
    airlock_free = 0
    cycles = []
    maxlen = max((len(v) for v in floater_jobs.values()), default=0)
    for k in range(maxlen):
        need = []
        for f in floaters:
            if k < len(floater_jobs[f]):
                jid = floater_jobs[f][k]
                g = jobs[jid - 1][0]
                if g > cur_contam[f]:
                    need.append(f)
        i = 0
        while i < len(need):
            group = need[i:i + C]
            start = max(airlock_free, max(cur_time[f] for f in group))
            end = start + T
            cycles.append(start)
            cid = len(cycles)
            for f in group:
                robot_events[f].append(("D", cid))
                cur_time[f] = end
                cur_contam[f] = L
            airlock_free = end
            i += C
        for f in floaters:
            if k < len(floater_jobs[f]):
                jid = floater_jobs[f][k]
                gg, rel, dur, dl, w = jobs[jid - 1]
                s = max(cur_time[f], rel)
                robot_events[f].append(("J", jid, s))
                finish = s + dur
                cur_time[f] = finish
                cur_contam[f] = gg

    out = []
    out.append(str(len(cycles)))
    for s in cycles:
        out.append(str(s))
    for r in range(1, R + 1):
        evs = robot_events[r]
        out.append(f"ROBOT {r} {len(evs)}")
        for e in evs:
            if e[0] == "J":
                out.append(f"J {e[1]} {e[2]}")
            else:
                out.append(f"D {e[1]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
