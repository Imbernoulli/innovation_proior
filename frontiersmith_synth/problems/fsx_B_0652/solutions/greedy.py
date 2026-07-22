# TIER: greedy
"""Obvious two-arm recipe: sort all parts by deadline (EDF), and for each part
hand it to whichever ELIGIBLE arm would personally finish it soonest (ignoring
that the zone is a single shared resource -- the choice never anticipates a
collision). Only AFTER committing to that arm does it reactively push the
departure past a conflicting already-committed zone job of the other arm (a
minimal, purely local patch -- it never reconsiders WHICH arm should have
taken the part, or re-orders anything to make room), dropping the part if the
push makes it miss its own deadline. Both arms end up racing for the same
soonest-expiring, item-rich central region and spend a lot of ticks yielding
to each other instead of trading off the zone by design."""
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


def main():
    W, zlo, zhi, posL0, posR0, n, items, TL, TR = read_input()

    state = {
        'L': {'node': 0, 'pos': posL0, 'time': 0, 'zones': [], 'order': []},
        'R': {'node': 1, 'pos': posR0, 'time': 0, 'zones': [], 'order': []},
    }
    tables = {'L': TL, 'R': TR}
    total = 0

    order_idx = sorted(range(n), key=lambda i: (items[i][2], -items[i][3], i))

    for idx in order_idx:
        pos, t, e, v, pd = items[idx]
        cands = []
        if pos <= zhi:
            cands.append('L')
        if pos >= zlo:
            cands.append('R')
        if not cands:
            continue

        # myopic choice: whichever eligible arm finishes soonest, ignoring the
        # other arm's schedule entirely
        best_arm, best_finish = None, None
        for arm in cands:
            st = state[arm]
            table = tables[arm]
            item_node = 2 + idx
            move_time = table[st['node']][item_node]
            arrival = st['time'] + move_time
            pick_start = max(arrival, t)
            pick_end = pick_start + pd
            if best_finish is None or pick_end < best_finish:
                best_finish, best_arm = pick_end, arm

        arm = best_arm
        other = 'R' if arm == 'L' else 'L'
        st = state[arm]
        table = tables[arm]
        item_node = 2 + idx
        move_time = table[st['node']][item_node]
        prev_pos = st['pos']
        in_zone = (zlo <= pos <= zhi) or (zlo <= prev_pos <= zhi)

        job_start = st['time']
        if in_zone:
            # purely local patch: push past whatever the other arm has ALREADY
            # committed; never reconsiders the arm choice itself
            changed = True
            guard = 0
            while changed and guard < 1000:
                changed = False
                guard += 1
                arrival = job_start + move_time
                pick_start = max(arrival, t)
                pick_end = pick_start + pd
                for (s, en) in state[other]['zones']:
                    if job_start < en and s < pick_end:
                        job_start = en
                        changed = True
                        break
        arrival = job_start + move_time
        pick_start = max(arrival, t)
        pick_end = pick_start + pd

        if pick_end > e:
            continue  # dropped: missed its deadline even after the local patch

        if in_zone:
            st['zones'].append((job_start, pick_end))
        st['node'] = item_node
        st['pos'] = pos
        st['time'] = pick_end
        st['order'].append((idx, job_start))
        total += v

    def flat(pairs):
        out = []
        for idx, dep in pairs:
            out.append(str(idx))
            out.append(str(dep))
        return " ".join(out)

    print(len(state['L']['order']))
    print(flat(state['L']['order']))
    print(len(state['R']['order']))
    print(flat(state['R']['order']))


if __name__ == "__main__":
    main()
