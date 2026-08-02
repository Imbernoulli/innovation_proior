# TIER: strong
# Register-pressure-aware list scheduling. The insight: a slot being free and an
# op being data-ready is NOT sufficient reason to issue it. Every op with a
# future consumer opens a new live range; every op consumed retires part of an
# old one. At each cycle we compute, for every ready op, its NET effect on the
# live-value count (net_delta = 1 if it starts a new live range minus however
# many currently-live predecessors it fully retires). Ops with net_delta <= 0
# are always safe to issue (they never grow pressure). Ops with net_delta > 0
# (pure/partial producers) are admitted only while the PROJECTED live count
# stays within a conservative budget -- deliberately narrower than the raw
# register file size, because starting more independent producers than the
# slow, serial retiring path (the accumulator chain) can actually drain buys
# nothing: any producer's own compute time is short enough to hide inside the
# latency of the op that will eventually retire it, so declining to launch it
# early costs no cycles, only avoids a doomed-to-spill live range. A single
# "nothing is even in flight" fallback guarantees forward progress; it is the
# only case allowed to exceed the budget.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); W = int(next(it)); R = int(next(it)); spill_cost = int(next(it))
    slot_types = next(it)

    ops = [None]
    consumers = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        typ = next(it)
        lat = int(next(it))
        k = int(next(it))
        preds = [int(next(it)) for _ in range(k)]
        ops.append((typ, lat, preds))
        for p in preds:
            consumers[p].append(i)

    remaining_consumers = [len(c) for c in consumers]
    live = set()
    sched = {}
    unscheduled = set(range(1, N + 1))
    cycle = 1
    max_cycle_guard = 400 * N + 2000
    # Deliberately narrower than R: R is how many live values COULD fit, not how
    # many independent producer chains are worth having in flight at once when
    # the retiring path is serial and slow.
    cap = max(2, R - 1) if R >= 3 else R

    def retiring(x, live_set):
        return sum(1 for p in ops[x][2] if p in live_set and remaining_consumers[p] == 1)

    def will_produce(x):
        return 1 if remaining_consumers[x] > 0 else 0

    while unscheduled and cycle <= max_cycle_guard:
        ready = []
        for i in unscheduled:
            preds = ops[i][2]
            if preds:
                if not all(p in sched for p in preds):
                    continue
                bound = max(sched[p][0] + ops[p][1] for p in preds)
            else:
                bound = 1
            if bound <= cycle:
                ready.append(i)

        ready.sort(key=lambda x: (will_produce(x) - retiring(x, live), x))

        used_slots = set()
        progressed = False
        for x in ready:
            typ = ops[x][0]
            slot = None
            for s in range(W):
                if s in used_slots:
                    continue
                if slot_types[s] == typ:
                    slot = s
                    break
            if slot is None:
                continue
            proj = len(live) - retiring(x, live) + will_produce(x)
            if proj > cap:
                continue  # leave this slot deliberately empty this cycle
            used_slots.add(slot)
            sched[x] = (cycle, slot)
            unscheduled.discard(x)
            progressed = True
            for p in ops[x][2]:
                if p in live:
                    remaining_consumers[p] -= 1
                    if remaining_consumers[p] == 0:
                        live.discard(p)
            if remaining_consumers[x] > 0:
                live.add(x)

        if not progressed:
            # Nothing is even in flight anywhere -- true starvation, not just a
            # latency gap that will resolve on its own. Force exactly one op
            # through so the schedule can never stall.
            any_in_flight = any(
                any(p in sched for p in ops[i][2])
                for i in unscheduled if ops[i][2]
            )
            if not any_in_flight:
                for x in ready:
                    typ = ops[x][0]
                    slot = None
                    for s in range(W):
                        if s in used_slots:
                            continue
                        if slot_types[s] == typ:
                            slot = s
                            break
                    if slot is None:
                        continue
                    used_slots.add(slot)
                    sched[x] = (cycle, slot)
                    unscheduled.discard(x)
                    for p in ops[x][2]:
                        if p in live:
                            remaining_consumers[p] -= 1
                            if remaining_consumers[p] == 0:
                                live.discard(p)
                    if remaining_consumers[x] > 0:
                        live.add(x)
                    break

        cycle += 1

    out = ["%d %d" % sched[i] for i in range(1, N + 1)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
