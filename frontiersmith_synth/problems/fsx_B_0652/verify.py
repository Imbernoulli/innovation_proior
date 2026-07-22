#!/usr/bin/env python3
"""Checker for 'conveyor-dual-arm-expiry'.

Usage: python3 verify.py <in> <out> <ans>   (ans unused)
Prints '... Ratio: <float in [0,1]>' on its final line and exits 0.
"""
import sys


def fail(reason):
    print("INVALID: %s Ratio: 0.0" % reason)
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    W = int(nx()); zlo = int(nx()); zhi = int(nx())
    posL0 = int(nx()); posR0 = int(nx())
    n = int(nx())
    items = []  # (pos,t,e,value,pickdur)
    for _ in range(n):
        pos = int(nx()); t = int(nx()); e = int(nx()); v = int(nx()); pd = int(nx())
        items.append((pos, t, e, v, pd))
    m = n + 2
    TL = [[int(nx()) for _ in range(m)] for _ in range(m)]
    TR = [[int(nx()) for _ in range(m)] for _ in range(m)]
    return W, zlo, zhi, posL0, posR0, n, items, TL, TR


def read_output(path, n):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output file")
    it = iter(toks)

    def nx_int():
        tok = next(it)
        v = int(tok)  # raises ValueError on 'nan'/'inf'/garbage -> caught by caller
        return v

    try:
        kL = nx_int()
        if kL < 0 or kL > n:
            fail("kL out of range")
        listL = [(nx_int(), nx_int()) for _ in range(kL)]
        kR = nx_int()
        if kR < 0 or kR > n:
            fail("kR out of range")
        listR = [(nx_int(), nx_int()) for _ in range(kR)]
    except StopIteration:
        fail("output truncated")
    except ValueError:
        fail("non-integer token in output")
    return listL, listR


def simulate_arm(items, table, home_node, is_left, zlo, zhi, index_list, seen):
    """Replay one arm's committed (item, depart_time) list. The solver chooses
    depart_time explicitly (it may be delayed past the arm's own free time to
    deliberately dodge the shared zone) -- that is the only way a plan can
    schedule the mutex. Returns (total_value, zone_intervals) or calls fail()."""
    cur_node = home_node
    cur_time = 0
    total_value = 0
    zone_intervals = []
    prev_pos = _HOME_POS[home_node]
    for (idx, depart_time) in index_list:
        if idx < 0 or idx >= len(items):
            fail("item index out of range")
        if idx in seen:
            fail("item %d listed more than once" % idx)
        seen.add(idx)
        if depart_time < 0 or depart_time < cur_time:
            fail("item %d: depart_time %d before arm is free (%d)" % (idx, depart_time, cur_time))
        pos, t, e, v, pd = items[idx]
        if is_left:
            if pos > zhi:
                fail("item %d unreachable by arm L" % idx)
        else:
            if pos < zlo:
                fail("item %d unreachable by arm R" % idx)
        item_node = 2 + idx
        move_time = table[cur_node][item_node]
        arrival = depart_time + move_time
        pick_start = arrival if arrival > t else t
        pick_end = pick_start + pd
        if pick_end > e:
            fail("item %d expires before pick completes" % idx)
        in_zone = (zlo <= pos <= zhi) or (zlo <= prev_pos <= zhi)
        job_start = depart_time
        job_end = pick_end
        if in_zone:
            zone_intervals.append((job_start, job_end))
        total_value += v
        cur_node = item_node
        cur_time = job_end
        prev_pos = pos
    return total_value, zone_intervals


_HOME_POS = {}  # node 0 -> posL0, node 1 -> posR0 (set in main)


def intervals_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def edf_single_arm(items, table, home_node, is_left, zlo, zhi):
    """Checker's own (deliberately trivial) baseline: ONE arm, EDF order, and it
    doesn't even engage with the shared zone -- it only services its own
    exclusive flank. This is the "don't bother scheduling the mutex at all"
    reference; any plan that actually uses the zone (with or without skill)
    should clear it comfortably."""
    reach = []
    for idx, (pos, t, e, v, pd) in enumerate(items):
        if is_left and pos >= zlo:
            continue
        if (not is_left) and pos <= zhi:
            continue
        reach.append(idx)
    reach.sort(key=lambda i: (items[i][2], -items[i][3], i))
    cur_node = home_node
    cur_time = 0
    total = 0
    for idx in reach:
        pos, t, e, v, pd = items[idx]
        item_node = 2 + idx
        move_time = table[cur_node][item_node]
        arrival = cur_time + move_time
        pick_start = arrival if arrival > t else t
        pick_end = pick_start + pd
        if pick_end <= e:
            total += v
            cur_node = item_node
            cur_time = pick_end
    return total


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    W, zlo, zhi, posL0, posR0, n, items, TL, TR = read_input(in_path)
    _HOME_POS[0] = posL0
    _HOME_POS[1] = posR0

    listL, listR = read_output(out_path, n)

    seen = set()
    valL, zonesL = simulate_arm(items, TL, 0, True, zlo, zhi, listL, seen)
    valR, zonesR = simulate_arm(items, TR, 1, False, zlo, zhi, listR, seen)

    for a in zonesL:
        for b in zonesR:
            if intervals_overlap(a, b):
                fail("zone mutex violated (arm L job %s overlaps arm R job %s)" % (a, b))

    F = valL + valR

    B_L = edf_single_arm(items, TL, 0, True, zlo, zhi)
    B_R = edf_single_arm(items, TR, 1, False, zlo, zhi)
    B = max(B_L, B_R)
    if B <= 0:
        B = 1  # defensive; generator guarantees a reachable freebie so this shouldn't trigger

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
