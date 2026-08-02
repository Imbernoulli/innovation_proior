#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE wire-schema-evolution instance to stdout.
Deterministic: all randomness seeded from testId only.

Instance shape: a service ships V wire-protocol versions. Every field, once
introduced at some version, stays in the schema forever (fields are never
retired -- old stored/replicated records keep referencing them). Each field
has a constant per-record occurrence "freq" for every version it is active
in. Fields are organized into thematic groups purely for flavor; the cost
model does not depend on group id.

Output format:
    line 1: V M T1CAP T2CAP T2COST T3COST
    next M lines: field_id group_id v0 freq   (field_id = 0..M-1, in the
        order the schema evolved -- version 1 fields first, then version 2's
        additions, etc. This "arrival order" is exactly the order a team
        would encounter fields while shipping the service.)
"""
import sys, random

T1CAP = 10     # tags [0, T1CAP)                  cost 1 byte/occurrence
T2CAP = 16     # tags [T1CAP, T1CAP+T2CAP)         cost 2 bytes/occurrence
T2COST = 2
T3COST = 12    # tags [T1CAP+T2CAP, inf)           cost T3COST bytes/occurrence (overflow key)

TRAP_TESTIDS = {4, 5, 7, 9, 10}


def gen_instance(test_id: int):
    rnd = random.Random(90173 + 7919 * test_id)
    V = 3 + (test_id % 3)          # 3..5 versions
    G = 3 + ((test_id + 1) % 3)    # 3..5 field groups
    bg_lo, bg_hi = 1, 120

    fields = []  # (group, v0, freq) in arrival order

    for g in range(G):
        nbase = rnd.randint(6, 10)
        for _ in range(nbase):
            freq = rnd.randint(bg_lo, bg_hi)
            fields.append((g, 1, freq))

    # Plant the trap: on trap testIds, one (or two, on the hardest case) group
    # gets a field that is only introduced near the LAST version, but at a
    # frequency 30x-120x anything seen at version 1. A tag-assignment policy
    # that commits its cheap tags using only what it can see at version 1
    # (or version-by-version, never revisiting) cannot reserve room for it.
    is_trap = test_id in TRAP_TESTIDS
    trap_slots = set()
    if is_trap:
        n_traps = 2 if test_id == 10 else 1
        for _ in range(n_traps):
            g = rnd.randint(0, G - 1)
            v_spike = rnd.choice([max(2, V - 1), V])
            trap_slots.add((g, v_spike))

    for v in range(2, V + 1):
        for g in range(G):
            n_grow = rnd.randint(2, 4)
            for _ in range(n_grow):
                freq = rnd.randint(bg_lo, bg_hi)
                fields.append((g, v, freq))
            if (g, v) in trap_slots:
                mult = rnd.randint(30, 120)
                freq = bg_hi * mult
                fields.append((g, v, freq))

    # background size/noise growth with testId (small -> large ladder)
    for _ in range(test_id):
        v0 = rnd.randint(1, V)
        freq = rnd.randint(bg_lo, bg_hi)
        g = rnd.randint(0, G - 1)
        fields.append((g, v0, freq))

    return V, fields


def main():
    test_id = int(sys.argv[1])
    V, fields = gen_instance(test_id)
    M = len(fields)
    out = [f"{V} {M} {T1CAP} {T2CAP} {T2COST} {T3COST}"]
    for fid, (g, v0, freq) in enumerate(fields):
        out.append(f"{fid} {g} {v0} {freq}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
