#!/usr/bin/env python3
"""gen.py <testId> -- deterministic instance generator for fsx_A_1364
(dual-quota / matroid-intersection-certify).

testId 1..3  : ALIGNED warm-ups. Scheme B (M2) never binds (every item has its own
               private scheme-B class), so the whole instance reduces to a single
               partition matroid (Scheme A) -- any sane construction, including the
               "obvious" single-pass greedy, reaches the true optimum here.
testId 4..10 : CONFLICT cases built from repeated "mega-gadgets" that are a planted
               instance of the classical fact that one-pass (no-exchange) common-
               independent-set construction can strand itself far from optimal:
               a bridge item ('b') that is processed first grabs a slot in BOTH an
               A-class it shares with 'a' and a B-class it shares with 'c' and a
               whole family of "wasted-slot" items p_1..p_m (m=2), permanently
               blocking the M1-only-first baseline from ever reaching the q_i's
               whose A-class becomes unreachable once p_i (uselessly) claims it.
"""
import sys


def build_aligned(test_id):
    configs = {
        1: [3, 3, 3],
        2: [3, 3, 3, 3],
        3: [4, 3, 3, 3, 3],
    }
    sizes = configs[test_id]
    K1 = len(sizes)
    cap1 = [2] * K1
    items = []
    next_b = 1
    for c in range(K1):
        for _ in range(sizes[c]):
            items.append((c + 1, next_b))
            next_b += 1
    K2 = next_b - 1
    cap2 = [1] * K2
    return items, K1, K2, cap1, cap2


def build_conflict(test_id):
    # K = number of mega-gadget copies; m = hub width inside each copy.
    K = test_id - 2  # test 4 -> K=2 ... test 10 -> K=8
    m = 2
    items = []
    cap1 = []
    cap2 = []

    def new1():
        cap1.append(1)
        return len(cap1)  # 1-indexed class id

    def new2():
        cap2.append(1)
        return len(cap2)

    for _copy in range(K):
        BX = new1()          # M1 class shared by a, b
        CX = new1()          # M1 class private to c
        AY = new2()          # M2 class private to a
        BY = new2()          # M2 class shared by b, c, and every p_i
        Gs = [new1() for _ in range(m)]   # M1 class shared by (p_i, q_i)
        Qs = [new2() for _ in range(m)]   # M2 class private to q_i

        items.append((BX, BY))            # b  (placed FIRST: the trap trigger)
        items.append((BX, AY))            # a
        items.append((CX, BY))            # c
        for i in range(m):
            items.append((Gs[i], BY))     # p_i  (a decoy: shares crowded BY)
        for i in range(m):
            items.append((Gs[i], Qs[i]))  # q_i  (the item baseline can never reach)

    K1 = len(cap1)
    K2 = len(cap2)
    return items, K1, K2, cap1, cap2


def main():
    test_id = int(sys.argv[1])
    if test_id <= 3:
        items, K1, K2, cap1, cap2 = build_aligned(test_id)
    else:
        items, K1, K2, cap1, cap2 = build_conflict(test_id)

    n = len(items)
    out = [f"{n} {K1} {K2}"]
    for c1, c2 in items:
        out.append(f"{c1} {c2}")
    out.append(" ".join(map(str, cap1)))
    out.append(" ".join(map(str, cap2)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
