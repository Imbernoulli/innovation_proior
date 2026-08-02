# TIER: trivial
# Reproduce the checker's own baseline construction exactly: one op issued per
# cycle, strictly in DAG (index) order, into the first slot of matching type.
# No bundle-packing, no pressure-awareness -- this is the "do nothing clever" ref.
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

    type_slot = {}
    for si, t in enumerate(slot_types):
        type_slot.setdefault(t, si)

    cyc = [0] * (N + 1)
    last_used = 0
    out = []
    for i in range(1, N + 1):
        typ, lat, preds = ops[i]
        bound = 1
        for p in preds:
            bound = max(bound, cyc[p] + ops[p][1])
        c = max(bound, last_used + 1)
        cyc[i] = c
        last_used = c
        out.append("%d %d" % (c, type_slot[typ]))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
