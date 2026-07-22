# TIER: strong
"""Insight: water couples every cycle globally, so the true scarce quantity is the
NUMBER OF CYCLES -- above all, the number of expensive direction SWITCHES -- the
shared reservoir can afford, not per-barge delay. Chasing whichever barge is most
urgent across both directions (the greedy tier) switches direction almost every
cycle and drains the reservoir long before the fleet is through.

So: budget the schedule from the water supply FIRST, then pack barges into it.
  1. Estimate the cycles needed per direction if perfectly batched to capacity:
     K_needed = ceil(|dir0|/C) + ceil(|dir1|/C).
  2. A schedule with S switches among K cycles costs about (K-S)*ws + S*wa. Solve
     for the largest switch budget the reservoir could ever afford:
         S_max = floor((W0 - K_needed*ws) / (wa - ws)), clipped to [0, K_needed-1].
  3. Rather than blindly spending the whole budget (a few well-placed switches
     often beat many small ones -- extra switching that isn't needed just burns
     water for no benefit), simulate a SMALL FAMILY of skeletons -- 1, 2, ...,
     S_max+1 direction-blocks, evenly spread over the K_needed cycles -- and pick
     whichever realized schedule scores best under the checker's own objective
     (stranding penalty + weighted tardiness + water spent). Within each
     candidate, every cycle batches the current direction's already-arrived
     backlog to full capacity, earliest-due first.
  4. While simulating any candidate, self-check the exact reservoir/horizon rule
     the checker uses and stop admitting a cycle the instant it would be
     infeasible.

This spends the water budget on a small, PLANNED number of switches (chosen by
comparing a handful of realized skeletons) instead of one per arrival, so on the
trap instances it keeps almost the whole fleet moving while the myopic due-date
chaser has already run dry."""
import sys


def ceil_div(x, y):
    return (x + y - 1) // y if y else 0


def simulate_skeleton(n, C, L, t, H, W0, rho, ws, wa, barges, dir_lists, start_dir, chunk_cycles):
    """Round-robin direction-blocks of `chunk_cycles` cycles each, alternating,
    batching the current direction's arrived backlog to full capacity
    (earliest-due first). Returns (cycles, F) where F mirrors the checker's
    objective exactly."""
    pending = {0: set(dir_lists[0]), 1: set(dir_lists[1])}
    prev_end = 0
    prev_dir = None
    W = W0
    cycles = []
    cur_dir = start_dir
    chunk_left = chunk_cycles
    guard = 0
    while (pending[0] or pending[1]) and guard < 8 * n + 50:
        guard += 1
        avail_cur = [i for i in pending[cur_dir] if barges[i][0] <= prev_end]
        avail_other = [i for i in pending[1 - cur_dir] if barges[i][0] <= prev_end]
        if not avail_cur and avail_other:
            cur_dir = 1 - cur_dir
            chunk_left = chunk_cycles
            avail_cur, avail_other = avail_other, avail_cur

        if not avail_cur and not avail_other:
            future = [barges[i][0] for d2 in (0, 1) for i in pending[d2]]
            if not future:
                break
            prev_end = max(prev_end, min(future))
            continue

        d = cur_dir if avail_cur else (1 - cur_dir)
        pool = pending[d]
        avail = sorted((i for i in pool if barges[i][0] <= prev_end),
                       key=lambda i: (barges[i][3], i))
        batch = []
        tot_len = 0
        for i in avail:
            if len(batch) >= C:
                break
            ln = barges[i][2]
            if tot_len + ln <= L:
                batch.append(i)
                tot_len += ln
        if not batch:
            break
        s = prev_end
        if s + t > H:
            break
        cost = ws if (prev_dir is None or d == prev_dir) else wa
        dt = s - prev_end  # back-to-back here, always 0
        W_after = W + rho * dt - cost
        if W_after < 0:
            break
        W = W_after
        cycles.append((s, d, [i + 1 for i in batch]))
        for i in batch:
            pending[d].discard(i)
        prev_end = s + t
        prev_dir = d

        chunk_left -= 1
        if chunk_left <= 0:
            cur_dir = 1 - cur_dir
            chunk_left = chunk_cycles

    finish = {}
    water_used = 0
    for (s, dd, idxs) in cycles:
        for i in idxs:
            finish[i - 1] = s + t
    prev_end2 = 0
    prev_dir2 = None
    for (s, dd, idxs) in cycles:
        cost = ws if (prev_dir2 is None or dd == prev_dir2) else wa
        water_used += cost
        prev_dir2 = dd
    F = water_used
    PEN = H
    for i in range(n):
        a, d, ln, due, wt = barges[i]
        if i in finish:
            F += wt * max(0, finish[i] - due)
        else:
            F += wt * PEN
    return cycles, F


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); C = int(next(it)); L = int(next(it)); t = int(next(it))
    H = int(next(it)); W0 = int(next(it)); rho = int(next(it))
    ws = int(next(it)); wa = int(next(it))
    barges = []
    for i in range(n):
        a = int(next(it)); d = int(next(it)); ln = int(next(it))
        due = int(next(it)); wt = int(next(it))
        barges.append((a, d, ln, due, wt))

    dir_lists = {0: [i for i in range(n) if barges[i][1] == 0],
                 1: [i for i in range(n) if barges[i][1] == 1]}

    K_needed = ceil_div(len(dir_lists[0]), C) + ceil_div(len(dir_lists[1]), C)
    K_needed = max(K_needed, 1)

    if wa > ws:
        S_max = (W0 - K_needed * ws) // (wa - ws)
    else:
        S_max = K_needed - 1
    S_max = max(0, min(S_max, K_needed - 1))

    def min_due(lst):
        return min((barges[i][3] for i in lst), default=1 << 60)

    start_dir = 0 if min_due(dir_lists[0]) <= min_due(dir_lists[1]) else 1

    best_cycles, best_F = None, None
    for num_blocks in range(1, S_max + 2):
        chunk_cycles = max(1, ceil_div(K_needed, num_blocks))
        cyc, F = simulate_skeleton(n, C, L, t, H, W0, rho, ws, wa, barges,
                                    dir_lists, start_dir, chunk_cycles)
        if best_F is None or F < best_F:
            best_F, best_cycles = F, cyc

    out = [str(len(best_cycles))]
    for (s, dd, idxs) in best_cycles:
        out.append("%d %d %d %s" % (s, dd, len(idxs), " ".join(map(str, idxs))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
