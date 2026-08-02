# TIER: greedy
# Max-fill list scheduling: at every cycle, pack every bundle slot with any op
# that is data-ready, picking the lowest-index ready op for each slot. This
# maximizes slot utilization / minimizes makespan and completely ignores how
# many values end up simultaneously live -- the "obvious" textbook recipe.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); W = int(next(it)); R = int(next(it)); spill_cost = int(next(it))
    slot_types = next(it)

    ops = [None]
    for i in range(1, N + 1):
        typ = next(it)
        lat = int(next(it))
        k = int(next(it))
        preds = [int(next(it)) for _ in range(k)]
        ops.append((typ, lat, preds))

    unscheduled = set(range(1, N + 1))
    sched = {}
    cycle = 1
    max_cycle_guard = 50 * N + 100

    while unscheduled and cycle <= max_cycle_guard:
        ready = []
        for i in unscheduled:
            preds = ops[i][2]
            bound = 1
            if preds:
                bound = max(sched[p][0] + ops[p][1] for p in preds) if all(p in sched for p in preds) else None
            if bound is not None and bound <= cycle:
                ready.append(i)
        ready.sort()

        free_slots = list(range(W))
        used_slots = set()
        for i in ready:
            typ = ops[i][0]
            slot = None
            for s in free_slots:
                if s in used_slots:
                    continue
                if slot_types[s] == typ:
                    slot = s
                    break
            if slot is None:
                continue
            used_slots.add(slot)
            sched[i] = (cycle, slot)
            unscheduled.discard(i)

        cycle += 1

    out = ["%d %d" % sched[i] for i in range(1, N + 1)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
