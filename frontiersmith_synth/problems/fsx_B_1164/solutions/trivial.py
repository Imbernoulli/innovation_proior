# TIER: trivial
"""Round-robin job assignment, completely ignoring grade; whenever a robot's
next job needs a higher grade than it currently has, fire an individual decon
cycle right away (never batched with anyone else). This mirrors the checker's
own internal naive baseline construction exactly."""
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

    order_jobs = sorted(range(1, J + 1), key=lambda j: (jobs[j - 1][1], j))

    cur_time = [0] * (R + 1)
    cur_contam = [L] * (R + 1)
    airlock_free = 0
    cycles = []              # list of start times
    robot_events = {r: [] for r in range(1, R + 1)}

    for idx, jid in enumerate(order_jobs):
        r = (idx % R) + 1
        g, rel, dur, dl, w = jobs[jid - 1]
        t = cur_time[r]
        if g > cur_contam[r]:
            start = max(airlock_free, t)
            end = start + T
            cycles.append(start)
            cid = len(cycles)
            airlock_free = end
            robot_events[r].append(("D", cid))
            cur_time[r] = end
            cur_contam[r] = L
            t = end
        s = max(t, rel)
        robot_events[r].append(("J", jid, s))
        finish = s + dur
        cur_time[r] = finish
        cur_contam[r] = g

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
