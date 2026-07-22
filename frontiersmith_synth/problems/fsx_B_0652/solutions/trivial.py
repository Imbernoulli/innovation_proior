# TIER: trivial
"""Do-nothing-clever baseline: use ONE arm only, earliest-deadline-first, and
never even engage the shared zone -- service only that arm's own exclusive
flank. Exactly reproduces the checker's own internal reference construction
(so F == B)."""
import sys


def read_input():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nx():
        return int(next(it))

    W = nx(); zlo = nx(); zhi = nx()
    posL0 = nx(); posR0 = nx()
    n = nx()
    items = []
    for _ in range(n):
        pos = nx(); t = nx(); e = nx(); v = nx(); pd = nx()
        items.append((pos, t, e, v, pd))
    m = n + 2
    TL = [[nx() for _ in range(m)] for _ in range(m)]
    TR = [[nx() for _ in range(m)] for _ in range(m)]
    return W, zlo, zhi, posL0, posR0, n, items, TL, TR


def edf_single_arm(items, table, home_node, is_left, zlo, zhi):
    reach = []
    for idx, (pos, t, e, v, pd) in enumerate(items):
        if is_left and pos >= zlo:
            continue  # never enters the shared zone
        if (not is_left) and pos <= zhi:
            continue
        reach.append(idx)
    reach.sort(key=lambda i: (items[i][2], -items[i][3], i))
    cur_node = home_node
    cur_time = 0
    total = 0
    order = []  # (idx, depart_time) -- alone, no mutex rival, so always depart ASAP
    for idx in reach:
        pos, t, e, v, pd = items[idx]
        item_node = 2 + idx
        move_time = table[cur_node][item_node]
        arrival = cur_time + move_time
        pick_start = arrival if arrival > t else t
        pick_end = pick_start + pd
        if pick_end <= e:
            total += v
            order.append((idx, cur_time))
            cur_node = item_node
            cur_time = pick_end
    return total, order


def flat(pairs):
    out = []
    for idx, dep in pairs:
        out.append(str(idx))
        out.append(str(dep))
    return " ".join(out)


def main():
    W, zlo, zhi, posL0, posR0, n, items, TL, TR = read_input()
    valL, orderL = edf_single_arm(items, TL, 0, True, zlo, zhi)
    valR, orderR = edf_single_arm(items, TR, 1, False, zlo, zhi)
    if valL >= valR:
        listL, listR = orderL, []
    else:
        listL, listR = [], orderR
    print(len(listL))
    print(flat(listL))
    print(len(listR))
    print(flat(listR))


if __name__ == "__main__":
    main()
