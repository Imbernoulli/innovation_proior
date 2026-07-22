# TIER: strong
"""Insight: the zone is a single shared resource, so the thing to schedule is
WHO holds it and WHEN, not which part to chase next. Builds several candidate
plans that each embody a facet of that idea -- an alternating zone cadence (in
both phases), a one-step mutex-aware lookahead, an exact brute-force
optimizer over each self-detected contention cluster, and a fully-specialized
split (one arm camps the zone, the other owns its own flank outright) -- keeps
whichever scores highest, then runs an exchange-argument repair sweep (flip
any zone part to its other eligible arm whenever that raises the total) on
top. Exclusive flank parts are always forced to their only reachable arm."""
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


def simulate(assign, items, TL, TR, posL0, posR0, zlo, zhi):
    """assign: dict idx -> 'L'/'R' (pre-decided). Commits in global EDF order,
    with the same reactive mutex-delay-then-drop-if-late replay as the checker."""
    state = {
        'L': {'node': 0, 'pos': posL0, 'time': 0, 'zones': [], 'order': []},
        'R': {'node': 1, 'pos': posR0, 'time': 0, 'zones': [], 'order': []},
    }
    tables = {'L': TL, 'R': TR}
    total = 0
    dropped = set()

    order_idx = sorted(assign.keys(), key=lambda i: (items[i][2], -items[i][3], i))
    for idx in order_idx:
        pos, t, e, v, pd = items[idx]
        arm = assign[idx]
        other = 'R' if arm == 'L' else 'L'
        st = state[arm]
        table = tables[arm]
        item_node = 2 + idx
        move_time = table[st['node']][item_node]
        prev_pos = st['pos']
        in_zone = (zlo <= pos <= zhi) or (zlo <= prev_pos <= zhi)

        job_start = st['time']
        if in_zone:
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
            dropped.add(idx)
            continue

        if in_zone:
            st['zones'].append((job_start, pick_end))
        st['node'] = item_node
        st['pos'] = pos
        st['time'] = pick_end
        st['order'].append((idx, job_start))
        total += v

    return total, state, dropped


def predictive_plan(items, n, TL, TR, posL0, posR0, zlo, zhi):
    """A second facet of the same insight: instead of committing to whichever
    arm has the smaller RAW finish time (greedy's mistake), look one step
    ahead and commit to whichever arm has the smaller finish time AFTER
    accounting for the zone hand-off with the other arm's schedule so far --
    i.e. directly optimize the thing that actually matters (post-mutex
    completion), rather than a proxy that ignores the shared resource."""
    state = {
        'L': {'node': 0, 'pos': posL0, 'time': 0, 'zones': [], 'order': []},
        'R': {'node': 1, 'pos': posR0, 'time': 0, 'zones': [], 'order': []},
    }
    tables = {'L': TL, 'R': TR}
    total = 0
    order_idx = sorted(range(n), key=lambda i: (items[i][2], -items[i][3], i))

    def tentative(arm, idx):
        pos, t, e, v, pd = items[idx]
        st = state[arm]
        other = 'R' if arm == 'L' else 'L'
        table = tables[arm]
        item_node = 2 + idx
        move_time = table[st['node']][item_node]
        prev_pos = st['pos']
        in_zone = (zlo <= pos <= zhi) or (zlo <= prev_pos <= zhi)
        job_start = st['time']
        if in_zone:
            changed, guard = True, 0
            while changed and guard < 1000:
                changed, guard = False, guard + 1
                arrival = job_start + move_time
                pick_start = arrival if arrival > t else t
                pick_end = pick_start + pd
                for (s, en) in state[other]['zones']:
                    if job_start < en and s < pick_end:
                        job_start, changed = en, True
                        break
        arrival = job_start + move_time
        pick_start = arrival if arrival > t else t
        pick_end = pick_start + pd
        return job_start, pick_end, in_zone, move_time

    assign = {}
    for idx in order_idx:
        pos, t, e, v, pd = items[idx]
        cands = []
        if pos <= zhi:
            cands.append('L')
        if pos >= zlo:
            cands.append('R')
        if not cands:
            continue
        best = None
        for arm in cands:
            job_start, pick_end, in_zone, move_time = tentative(arm, idx)
            key = (pick_end > e, pick_end)  # prefer feasible, then earliest finish
            if best is None or key < best[0]:
                best = (key, arm, job_start, pick_end, in_zone)
        _, arm, job_start, pick_end, in_zone = best
        assign[idx] = arm
        if pick_end > e:
            continue
        st = state[arm]
        item_node = 2 + idx
        if in_zone:
            st['zones'].append((job_start, pick_end))
        st['node'], st['pos'], st['time'] = item_node, pos, pick_end
        st['order'].append((idx, job_start))
        total += v
    return total, state, assign


def cluster_and_optimize(items, n, zlo, zhi, base_assign):
    """The sharpest facet of the insight: group zone parts into contention
    clusters purely from their OWN timing data (parts whose [t,e] windows sit
    close together in time are the ones that will actually fight over the
    zone), then brute-force the arm assignment within each small cluster --
    exact for the part of the problem where the mutex truly bites, instead of
    a single global heuristic rule."""
    zone_idx = [i for i in range(n) if zlo <= items[i][0] <= zhi]
    zone_idx.sort(key=lambda i: items[i][1])  # by appear time
    clusters = []
    cur = []
    last_t = None
    GAP = 6
    for i in zone_idx:
        t = items[i][1]
        if last_t is not None and t - last_t > GAP:
            clusters.append(cur)
            cur = []
        cur.append(i)
        last_t = t
    if cur:
        clusters.append(cur)

    assign = dict(base_assign)
    for cluster in clusters:
        k = len(cluster)
        if k == 0 or k > 14:
            continue  # too big to brute force; leave the heuristic assignment
        cluster_sorted = sorted(cluster, key=lambda i: (items[i][2], -items[i][3], i))
        best_val, best_combo = -1, None
        for mask in range(1 << k):
            local_state = {'L': {'time': 0, 'zones': []}, 'R': {'time': 0, 'zones': []}}
            val = 0
            for bit, idx in enumerate(cluster_sorted):
                arm = 'L' if (mask >> bit) & 1 == 0 else 'R'
                other = 'R' if arm == 'L' else 'L'
                pos, t, e, v, pd = items[idx]
                st = local_state[arm]
                job_start = st['time']
                changed, guard = True, 0
                while changed and guard < 200:
                    changed, guard = False, guard + 1
                    pick_start = job_start if job_start > t else t
                    pick_end = pick_start + pd
                    for (s, en) in local_state[other]['zones']:
                        if job_start < en and s < pick_end:
                            job_start, changed = en, True
                            break
                pick_start = job_start if job_start > t else t
                pick_end = pick_start + pd
                if pick_end <= e:
                    local_state[arm]['zones'].append((job_start, pick_end))
                    local_state[arm]['time'] = pick_end
                    val += v
            if val > best_val:
                best_val, best_combo = val, mask
        for bit, idx in enumerate(cluster_sorted):
            assign[idx] = 'L' if (best_combo >> bit) & 1 == 0 else 'R'
    return assign


def value_first_plan(items, n, TL, TR, posL0, posR0, zlo, zhi):
    """A third facet: visit parts by VALUE, not urgency. Earliest-deadline
    order is what greedy uses -- it dutifully drains cheap, urgent parts
    first and only reaches the big prizes late (often too late). Committing
    to high-value parts first, and only mopping up cheap filler with whatever
    arm-time is left, sidesteps that trap entirely."""
    state = {
        'L': {'node': 0, 'pos': posL0, 'time': 0, 'zones': [], 'order': []},
        'R': {'node': 1, 'pos': posR0, 'time': 0, 'zones': [], 'order': []},
    }
    tables = {'L': TL, 'R': TR}
    total = 0
    order_idx = sorted(range(n), key=lambda i: (-items[i][3], items[i][2], i))

    def tentative(arm, idx):
        pos, t, e, v, pd = items[idx]
        st = state[arm]
        other = 'R' if arm == 'L' else 'L'
        table = tables[arm]
        item_node = 2 + idx
        move_time = table[st['node']][item_node]
        prev_pos = st['pos']
        in_zone = (zlo <= pos <= zhi) or (zlo <= prev_pos <= zhi)
        job_start = st['time']
        if in_zone:
            changed, guard = True, 0
            while changed and guard < 1000:
                changed, guard = False, guard + 1
                arrival = job_start + move_time
                pick_start = arrival if arrival > t else t
                pick_end = pick_start + pd
                for (s, en) in state[other]['zones']:
                    if job_start < en and s < pick_end:
                        job_start, changed = en, True
                        break
        arrival = job_start + move_time
        pick_start = arrival if arrival > t else t
        pick_end = pick_start + pd
        return job_start, pick_end, in_zone

    assign = {}
    for idx in order_idx:
        pos, t, e, v, pd = items[idx]
        cands = []
        if pos <= zhi:
            cands.append('L')
        if pos >= zlo:
            cands.append('R')
        if not cands:
            continue
        best = None
        for arm in cands:
            job_start, pick_end, in_zone = tentative(arm, idx)
            key = (pick_end > e, pick_end)
            if best is None or key < best[0]:
                best = (key, arm, job_start, pick_end, in_zone)
        _, arm, job_start, pick_end, in_zone = best
        assign[idx] = arm
        if pick_end > e:
            continue
        st = state[arm]
        item_node = 2 + idx
        if in_zone:
            st['zones'].append((job_start, pick_end))
        st['node'], st['pos'], st['time'] = item_node, pos, pick_end
        st['order'].append((idx, job_start))
        total += v
    return total, state, assign


def build_specialist(items, n, zlo, zhi, specialist_arm):
    """The sharpest facet: dedicate ONE arm to camp the zone (it services
    every zone part back-to-back, never leaving, so it never re-pays travel
    to re-enter and never contends with the other arm at all) while the OTHER
    arm handles only its own exclusive flank. Costs that arm its own flank
    value, but eliminates both the mutex AND the switching-travel tax
    entirely for the zone throughput."""
    zone_idx = [i for i in range(n) if zlo <= items[i][0] <= zhi]
    other = 'R' if specialist_arm == 'L' else 'L'
    assign = {}
    for i in zone_idx:
        assign[i] = specialist_arm
    flank_own = [i for i in range(n)
                 if (items[i][0] < zlo and other == 'L') or (items[i][0] > zhi and other == 'R')]
    for i in flank_own:
        assign[i] = other
    return assign


def build_alternation(items, n, zlo, zhi, phase_first):
    zone_idx = [i for i in range(n) if zlo <= items[i][0] <= zhi]
    zone_idx.sort(key=lambda i: (items[i][2], -items[i][3], i))
    flankL = [i for i in range(n) if items[i][0] < zlo]
    flankR = [i for i in range(n) if items[i][0] > zhi]

    assign = {}
    for i in flankL:
        assign[i] = 'L'
    for i in flankR:
        assign[i] = 'R'
    turn = phase_first
    for i in zone_idx:
        assign[i] = turn
        turn = 'R' if turn == 'L' else 'L'
    return assign, zone_idx


def main():
    W, zlo, zhi, posL0, posR0, n, items, TL, TR = read_input()

    assignA, zone_idx = build_alternation(items, n, zlo, zhi, 'L')
    assignB, _ = build_alternation(items, n, zlo, zhi, 'R')

    totalA, stateA, droppedA = simulate(assignA, items, TL, TR, posL0, posR0, zlo, zhi)
    totalB, stateB, droppedB = simulate(assignB, items, TL, TR, posL0, posR0, zlo, zhi)
    totalC, stateC, assignC = predictive_plan(items, n, TL, TR, posL0, posR0, zlo, zhi)
    committedC = {idx for idx, _ in stateC['L']['order']} | {idx for idx, _ in stateC['R']['order']}
    droppedC = set(assignC.keys()) - committedC

    assignD = cluster_and_optimize(items, n, zlo, zhi, assignA)
    totalD, stateD, droppedD = simulate(assignD, items, TL, TR, posL0, posR0, zlo, zhi)

    assignE = build_specialist(items, n, zlo, zhi, 'L')
    totalE, stateE, droppedE = simulate(assignE, items, TL, TR, posL0, posR0, zlo, zhi)
    assignF = build_specialist(items, n, zlo, zhi, 'R')
    totalF, stateF, droppedF = simulate(assignF, items, TL, TR, posL0, posR0, zlo, zhi)

    totalG, stateG, assignG = value_first_plan(items, n, TL, TR, posL0, posR0, zlo, zhi)
    committedG = {idx for idx, _ in stateG['L']['order']} | {idx for idx, _ in stateG['R']['order']}
    droppedG = set(assignG.keys()) - committedG

    candidates = [
        (totalA, assignA, stateA, droppedA),
        (totalB, assignB, stateB, droppedB),
        (totalC, assignC, stateC, droppedC),
        (totalD, assignD, stateD, droppedD),
        (totalE, assignE, stateE, droppedE),
        (totalF, assignF, stateF, droppedF),
        (totalG, assignG, stateG, droppedG),
    ]
    candidates.sort(key=lambda c: -c[0])
    best_total, best_assign, best_state, best_dropped = candidates[0]

    # exchange-argument repair: flip a zone part to its other arm (zone parts
    # are reachable by both) whenever that raises the grand total. Sweep ALL
    # zone parts, not just currently-dropped ones -- a flip can free up zone
    # time that lets a DIFFERENT part succeed, a classic exchange argument.
    improved = True
    guard = 0
    while improved and guard < 40:
        improved = False
        guard += 1
        for idx in zone_idx:
            trial = dict(best_assign)
            trial[idx] = 'R' if best_assign.get(idx) == 'L' else 'L'
            t_total, t_state, t_dropped = simulate(trial, items, TL, TR, posL0, posR0, zlo, zhi)
            if t_total > best_total:
                best_assign, best_total, best_state, best_dropped = trial, t_total, t_state, t_dropped
                improved = True

    def flat(pairs):
        out = []
        for idx, dep in pairs:
            out.append(str(idx))
            out.append(str(dep))
        return " ".join(out)

    listL = best_state['L']['order']
    listR = best_state['R']['order']
    print(len(listL))
    print(flat(listL))
    print(len(listR))
    print(flat(listR))


if __name__ == "__main__":
    main()
