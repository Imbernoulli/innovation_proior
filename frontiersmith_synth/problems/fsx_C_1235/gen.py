#!/usr/bin/env python3
"""gen.py <testId> -- prints one migration-timeline instance to stdout.

Deterministic: testId fully determines the instance (an internal RNG is
seeded from testId only, used solely to pick cosmetic values, never
structure/timing).

Instance shape:
    line 1: "K T M"          K keys, T ticks (0-indexed 0..T-1), M backfill ticks
    line 2: "V_1 ... V_K"    baseline value of each key (as currently stored in
                              the OLD store, before migration starts; implicit
                              version 0)
    T lines, one per tick, in execution order:
      "L k v x"   LIVE dual-write: key k (1-indexed), new version v (v>=1,
                   strictly increasing per key), new value x. Dual-write always
                   lands correctly in the NEW store (no race on the live path).
      "B k v x"   BACKFILL tick: a background worker read key k from OLD at an
                   earlier snapshot instant, when its (version,value) was
                   exactly (v,x), and is only NOW, at this tick, writing that
                   pair into NEW. v is always some (version,value) key k truly
                   held earlier (0/baseline, or an earlier L on the same key).
      "R k"       READ-CHECK at key k.
"""
import sys, random

def build(test_id: int):
    rng = random.Random(1000 + test_id)

    # ---- per-test-id topology: (K, list of key roles) ----
    # roles: 'quiet_b'  -> single baseline backfill, safe
    #        'quiet_lb' -> live write then a matching (non-stale) backfill
    #        'quiet_l'  -> live write only, no backfill needed
    #        'trap'     -> L(v1) , L(v2) , B(snapshot=v1, STALE) race
    #        'gate'     -> an early read placed before this key's own (later) touch
    profiles = {
        1:  (3,  ['quiet_b', 'quiet_l', 'gate']),
        2:  (4,  ['quiet_b', 'quiet_lb', 'quiet_l', 'gate']),
        3:  (4,  ['trap', 'quiet_b', 'quiet_l', 'gate']),
        4:  (5,  ['quiet_b', 'quiet_lb', 'quiet_l', 'quiet_b', 'gate']),
        5:  (6,  ['trap', 'trap', 'quiet_b', 'quiet_lb', 'quiet_l', 'gate']),
        6:  (6,  ['quiet_b', 'quiet_lb', 'quiet_l', 'quiet_b', 'quiet_lb', 'gate']),
        7:  (8,  ['trap', 'trap', 'trap', 'quiet_b', 'quiet_lb', 'quiet_l', 'quiet_b', 'gate']),
        8:  (7,  ['quiet_b', 'quiet_lb', 'quiet_l', 'quiet_b', 'quiet_lb', 'quiet_l', 'gate']),
        9:  (9,  ['trap', 'trap', 'quiet_b', 'quiet_lb', 'quiet_l', 'quiet_b', 'quiet_lb', 'quiet_l', 'gate']),
        10: (10, ['trap', 'trap', 'trap', 'quiet_b', 'quiet_lb', 'quiet_l', 'quiet_b', 'quiet_lb', 'quiet_l', 'gate']),
    }
    K, roles = profiles[test_id]
    assert len(roles) == K

    baseline = [rng.randint(1, 200) for _ in range(K)]

    # segments assembled independently, then concatenated in a fixed order so
    # that (a) every trap key's 3 writes keep their required relative order,
    # (b) every trap key's read-check lands strictly after ALL backfill ticks
    #     in the whole instance (so a naive "cutover right after the backfill
    #     phase ends" plan is always exposed to it), and
    # (c) the gate key's read lands strictly before its own (later) touch, so
    #     no plan can safely cut over at tick 0.
    writes = []     # L/B ticks (backfill + live), built first
    mid_reads = []  # read-checks for quiet keys (placed right after writes)
    trap_reads = [] # read-checks for trap keys (placed at the very end)
    gate_read = []
    gate_touch_slot = []  # inserted into `writes` near its end

    ver = {k: 0 for k in range(1, K + 1)}  # last-used live version per key

    def live(k, delta_lo=1, delta_hi=1):
        ver[k] += 1
        val = rng.randint(1, 200)
        writes.append(('L', k, ver[k], val))
        return ver[k], val

    gate_key = None
    for idx, role in enumerate(roles):
        k = idx + 1
        if role == 'quiet_b':
            writes.append(('B', k, 0, baseline[k - 1]))
            mid_reads.append(('R', k))
        elif role == 'quiet_lb':
            v, val = live(k)
            writes.append(('B', k, v, val))  # snapshot == current true state: safe
            mid_reads.append(('R', k))
        elif role == 'quiet_l':
            live(k)
            mid_reads.append(('R', k))
        elif role == 'trap':
            v1, val1 = live(k)               # p1
            v2, val2 = live(k)                # p3 (bumps past v1)
            writes.append(('B', k, v1, val1))  # p2: STALE snapshot, applied late
            trap_reads.append(('R', k))        # p4: must see (v2,val2)
        elif role == 'gate':
            gate_key = k
            gate_read.append(('R', k))
            gate_touch_slot.append(('B', k, 0, baseline[k - 1]))
        else:
            raise ValueError(role)

    # Place the gate's read ~40% and its own touch ~78% of the way through the
    # write phase (NOT at the very front) so the forced "no plan may cut over
    # before this tick" floor scales with the instance size T, instead of
    # collapsing to a fixed small constant on large instances.
    nw = len(writes)
    posA = max(1, (nw * 2) // 5)
    posB = max(posA + 1, (nw * 4) // 5)
    ops = []
    ops += writes[:posA]
    ops += gate_read
    ops += writes[posA:posB]
    ops += gate_touch_slot
    ops += writes[posB:]
    ops += mid_reads
    ops += trap_reads

    T = len(ops)
    M = sum(1 for o in ops if o[0] == 'B')
    return K, T, M, baseline, ops


def main():
    test_id = int(sys.argv[1])
    K, T, M, baseline, ops = build(test_id)
    out = [f"{K} {T} {M}", " ".join(str(v) for v in baseline)]
    for op in ops:
        if op[0] == 'R':
            out.append(f"R {op[1]}")
        else:
            out.append(f"{op[0]} {op[1]} {op[2]} {op[3]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
