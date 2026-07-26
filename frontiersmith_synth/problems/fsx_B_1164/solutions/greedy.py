# TIER: greedy
"""The 'obvious' first attempt: a myopic, reactive dispatcher. Process jobs
in release order. For each job, prefer a robot that is ALREADY the right
grade (no decon needed) among those currently free, breaking ties by whoever
freed up soonest; if no already-compatible robot is free yet, fall back to
whichever robot is free soonest overall and pay for an individual decon
cycle right then. This is the natural first fix a coder reaches for once they
notice decon is expensive -- "reuse a clean robot if you have one" -- but it
never plans ahead: it has no notion of dedicating robots to grades or of
batching several robots' decon needs together. Under a burst of many
same-grade jobs at once, every "already compatible" robot is instantly used
up and the dispatcher degrades to the same grade-blind, one-cycle-per-robot
scramble as picking whoever is free."""
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

    robot_free = [0] * (R + 1)
    robot_contam = [L] * (R + 1)
    airlock_free = 0
    cycles = []
    robot_events = {r: [] for r in range(1, R + 1)}
    robots = list(range(1, R + 1))

    for jid in order_jobs:
        g, rel, dur, dl, w = jobs[jid - 1]

        compat = [x for x in robots if robot_contam[x] >= g]
        compat_free_time = min((robot_free[x] for x in compat), default=None)
        earliest_free_time = min(robot_free[x] for x in robots)

        if compat_free_time is not None and compat_free_time <= max(earliest_free_time, rel):
            # some robot already covers this grade without a decon and is
            # available soon enough -- reuse it
            r = min(compat, key=lambda x: (robot_free[x], x))
        else:
            r = min(robots, key=lambda x: (robot_free[x], x))

        t = robot_free[r]
        if g > robot_contam[r]:
            start = max(airlock_free, t)
            end = start + T
            cycles.append(start)
            cid = len(cycles)
            airlock_free = end
            robot_events[r].append(("D", cid))
            robot_contam[r] = L
            t = end
        s = max(t, rel)
        robot_events[r].append(("J", jid, s))
        finish = s + dur
        robot_free[r] = finish
        robot_contam[r] = g

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
