#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  ->  prints 'Ratio: <x in [0,1]>' (last line authoritative).

Deterministic exact scorer for the lock-chamber green-wave problem.

Feasibility (checked strictly; ANY violation -> Ratio: 0.0):
  - output is exactly: K, then K cycles of "s dir m idx_1..idx_m" (no leftover tokens)
  - 0 <= K <= K_MAX; all tokens are finite integers
  - cycles strictly ordered: s_1 < s_2 < ... and s_j >= prev_end (no overlap), s_j+t <= H
  - 0 <= m_j <= C; sum of len over the batch <= L
  - every barge index in [1,n], used in AT MOST one cycle over the whole schedule
    (continuity: a barge transits at most once)
  - every barge assigned to cycle j has direction == dir_j and arrival <= s_j
  - the shared reservoir (start W0, +rho per IDLE tick before a cycle starts,
    -ws same-direction cost / -wa switch cost per cycle) must never go negative

Objective (minimize): F = sum over TRANSITED barges of wt_i * max(0, finish_i - due_i)
                         + sum over UNTRANSITED barges of wt_i * PEN,  PEN = H
                         + TOTAL_WATER_USED  (sum of the ws/wa actually paid)
(stranding a barge is deliberately far worse than any achievable lateness within H;
the water term keeps the shared reservoir inside the objective itself, not just a
feasibility gate, and guarantees F > 0 even for a flawless, on-time schedule).

Baseline B: the checker's own FCFS-with-batching construction -- repeatedly take
the earliest-arrived still-waiting barge, batch it (and same-direction companions,
earliest-arrived first) up to chamber capacity, and run that cycle if affordable;
stop the moment the next cycle is unaffordable or exceeds the horizon. It exploits
chamber-batching but has NO idea about the water budget or due dates -- the
"obvious" naive dispatcher the family's trap targets.

Minimization normalization: sc = min(1000, 100*B/max(1e-9,F)); Ratio = sc/1000.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it)); C = int(next(it)); L = int(next(it)); t = int(next(it))
    H = int(next(it)); W0 = int(next(it)); rho = int(next(it))
    ws = int(next(it)); wa = int(next(it))
    barges = []
    for i in range(n):
        a = int(next(it)); d = int(next(it)); ln = int(next(it))
        due = int(next(it)); wt = int(next(it))
        barges.append((a, d, ln, due, wt))
    return n, C, L, t, H, W0, rho, ws, wa, barges


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def simulate_baseline(n, C, L, t, H, W0, rho, ws, wa, barges):
    """FCFS-with-batching: repeatedly serve the direction of the earliest-arrived
    still-waiting barge, batching same-direction companions (earliest-arrived
    first) up to chamber capacity. No due-date or water-budget awareness at all
    -- stops permanently the moment the next cycle is unaffordable or would
    exceed the horizon."""
    unserved = set(range(n))
    prev_end = 0
    prev_dir = None
    W = W0
    finish = {}
    water_used = 0
    guard = 0
    while unserved and guard < 4 * n + 20:
        guard += 1
        cur_time = prev_end
        avail = [i for i in unserved if barges[i][0] <= cur_time]
        if not avail:
            nxt = min(barges[i][0] for i in unserved)
            cur_time = max(cur_time, nxt)
            avail = [i for i in unserved if barges[i][0] <= cur_time]
            if not avail:
                break
        avail.sort(key=lambda i: (barges[i][0], i))
        d = barges[avail[0]][1]
        cand = sorted((i for i in avail if barges[i][1] == d),
                      key=lambda i: (barges[i][0], i))
        batch = []
        tot_len = 0
        for i in cand:
            if len(batch) >= C:
                break
            ln = barges[i][2]
            if tot_len + ln <= L:
                batch.append(i)
                tot_len += ln
        if not batch:
            break
        s = cur_time
        if s + t > H:
            break
        cost = ws if (prev_dir is None or d == prev_dir) else wa
        dt = s - prev_end
        W_after = W + rho * dt - cost
        if W_after < 0:
            break
        W = W_after
        water_used += cost
        for i in batch:
            unserved.discard(i)
            finish[i] = s + t
        prev_end = s + t
        prev_dir = d
    F = water_used
    PEN = H
    for i in range(n):
        a, d, ln, due, wt = barges[i]
        if i in finish:
            F += wt * max(0, finish[i] - due)
        else:
            F += wt * PEN
    return F


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    n, C, L, t, H, W0, rho, ws, wa, barges = read_instance(inp)
    K_MAX = 10 * n + 20

    try:
        with open(outp) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    # --- strict, bounded-read integer parsing ---
    def next_int(pos):
        if pos >= len(raw):
            fail("unexpected end of output")
        tok = raw[pos]
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer token: %r" % tok)
        return v

    pos = 0
    K = next_int(pos); pos += 1
    if K < 0 or K > K_MAX:
        fail("K=%d out of bounds [0,%d]" % (K, K_MAX))

    cycles = []
    used = [False] * (n + 1)
    prev_end = 0
    prev_dir = None
    W = W0
    total_water_used = 0
    for j in range(K):
        s = next_int(pos); pos += 1
        d = next_int(pos); pos += 1
        m = next_int(pos); pos += 1
        if d not in (0, 1):
            fail("bad direction %d" % d)
        if m < 0 or m > C:
            fail("cycle %d batch size %d out of [0,%d]" % (j, m, C))
        idxs = []
        for _ in range(m):
            v = next_int(pos); pos += 1
            idxs.append(v)
        if s < 0 or s + t > H:
            fail("cycle %d start %d out of horizon [0,%d]" % (j, s, H))
        if j > 0 and s < prev_end:
            fail("cycle %d overlaps previous cycle (s=%d < prev_end=%d)" % (j, s, prev_end))

        tot_len = 0
        seen_here = set()
        for v in idxs:
            if v < 1 or v > n:
                fail("barge index %d out of [1,%d]" % (v, n))
            if v in seen_here:
                fail("barge %d repeated within cycle %d" % (v, j))
            seen_here.add(v)
            if used[v]:
                fail("barge %d assigned to more than one cycle" % v)
            used[v] = True
            a, bd, ln, due, wt = barges[v - 1]
            if bd != d:
                fail("barge %d direction %d != cycle direction %d" % (v, bd, d))
            if a > s:
                fail("barge %d arrives at %d after cycle start %d" % (v, a, s))
            tot_len += ln
        if tot_len > L:
            fail("cycle %d total length %d exceeds L=%d" % (j, tot_len, L))

        cost = ws if (prev_dir is None or d == prev_dir) else wa
        dt = s - prev_end
        W = W + rho * dt - cost
        if W < 0:
            fail("reservoir went negative at cycle %d (W=%d)" % (j, W))
        total_water_used += cost

        cycles.append((s, d, idxs))
        prev_end = s + t
        prev_dir = d

    if pos != len(raw):
        fail("trailing tokens after parsing %d cycles" % K)

    finish = {}
    for (s, d, idxs) in cycles:
        for v in idxs:
            finish[v - 1] = s + t

    PEN = H
    F = total_water_used
    for i in range(n):
        a, d, ln, due, wt = barges[i]
        if i in finish:
            F += wt * max(0, finish[i] - due)
        else:
            F += wt * PEN

    B = simulate_baseline(n, C, L, t, H, W0, rho, ws, wa, barges)
    B = max(B, 1)  # guard: B should always be positive here (PEN>0, n>=1)

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    n_transited = len(finish)
    sys.stdout.write(
        "F=%d B=%d transited=%d/%d\nRatio: %.6f\n" % (F, B, n_transited, n, ratio)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
