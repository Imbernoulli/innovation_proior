#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1126 -- "The Clerk Who Redacted Her Own Calendar"
(family: disguised-interval-allocation; format B, quality-metric).

THEME.  A boutique hotel's night clerk gets a stack of booking requests.  Two
bookings clash iff their stays overlap by at least one night; a clash means
they cannot share a room.  For a long-running privacy audit the hotel strips
every booking of its actual check-in/check-out dates before it reaches the
clerk -- she only ever sees a CLASH ROSTER (an anonymised conflict graph) and
each guest's contracted value.  She has `rooms` identical rooms and must seat
guests to minimise the total value of guests she turns away.  What she is not
told: every clash roster the hotel ever hands her is secretly an INTERVAL
GRAPH -- it came from real stay intervals on a single calendar, just with the
booking numbers reshuffled so the calendar order isn't visible.  A clerk who
notices this (and reconstructs the hidden day-by-day order) can solve
"who do I turn away" close to optimally; a clerk who just colours the graph
greedily, value by value, cannot.

MECHANISM 1 -- hidden-chordality-detection.  The public graph is chordal (in
fact an interval graph).  A candidate that runs Maximum Cardinality Search
(MCS) recovers a perfect elimination ordering; from it, maximal cliques and a
maximum-weight spanning tree of the clique-intersection graph recover a
*linear* clique order (the interval realization), turning an apparently
NP-hard weighted-graph-coloring-with-rejection problem into a scheduling
problem on a line.

MECHANISM 2 -- spill-placement-optimization.  Once a linear (interval) order
is available, deciding WHOM to turn away to respect the `rooms` capacity is a
resource-constrained weighted interval-scheduling problem: naive priority
(seat the highest-value guest first) can be beaten badly by an exchange (drop
one expensive guest whose stay blocks an entire room for its whole duration,
to free the room for many smaller, collectively more valuable, guests).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "rooms": R (int),
             "edges": [[i,j], ...],      # clash roster, 0 <= i<j<N
             "value": [v_0, ..., v_{N-1}]}   # integer value of each guest
  stdout: ONE JSON object:
            {"room": [r_0, ..., r_{N-1}]}
          where r_i is either an integer room index 0..R-1 (guest i is seated
          there) or -1 (guest i is turned away).

  A layout is VALID iff `room` has exactly N integer entries, each in
  {-1, 0, ..., R-1}, and for EVERY room no two seated guests clash (i.e. no
  edge of the roster has both endpoints in that room).  Invalid output, wrong
  length, an out-of-range index, a clash inside one room, a crash, a timeout,
  or non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Let `spill` = sum of value[i] over
turned-away guests.  Per instance we compute two references, both from data
the candidate never sees (the real hidden stay dates):
    base = spill of a WEAK reference clerk who seats guests in the (already
           reshuffled) roster order, first-fit into rooms 0..R-1 by the
           public clash graph alone,
    lb   = a LOOSE, generally UNREACHABLE lower bound on spill: at every day
           `t`, if `c(t)` guests are actually present and `c(t) > rooms`, ANY
           valid layout must turn away at least `c(t)-rooms` of the guests
           present at `t`; the cheapest possible way to do just that (ignoring
           every other day) costs `lb(t)`; lb = max over t of lb(t).  Because
           it only ever looks at one day at a time it never accounts for a
           single guest blocking several overloaded days at once, so it is
           strictly looser than the true optimum on most instances.
and normalize with an affine anchor (weak clerk -> 0.1, the loose ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (base - spill) / max(1e-9, base - lb), 0, 1 )
A candidate matching the weak clerk scores ~0.1; doing worse scores < 0.1;
approaching (but, because `lb` is loose, never quite reaching) the ideal
scores close to but below 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (roster +
values).  The hidden stay dates and both references are computed by THIS
parent process, so a frame-walking / introspecting candidate learns nothing.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance construction ------------------------
def _trap_window(ni, start, length, R, K, block_val, victim_val, jitter):
    """R 'blocker' stays spanning the whole window (value ~block_val each) plus
    R tracks of K short sequential 'victim' stays (value ~victim_val each,
    K*victim_val > block_val > victim_val) nested inside.  A value-first
    clerk seats every blocker (each individually outvalues any one victim),
    consuming all R rooms for the whole window, and so rejects EVERY victim --
    even though dropping the blockers and keeping their victim tracks is
    worth strictly more.  Returns list of (s, e, value)."""
    out = []
    end = start + length
    for _ in range(R):
        v = block_val + ni(-jitter, jitter)
        out.append((start, end, max(1, v)))
    for _t in range(R):
        cur = start
        for _k in range(K):
            remaining_slots = K - _k
            room_left = end - cur
            if room_left <= remaining_slots:
                break
            seg = max(2, room_left // (remaining_slots + 1))
            gap = ni(0, max(1, seg // 4))
            s = cur + gap
            e = s + max(2, seg - gap)
            if e >= end:
                e = end - 1
            if e <= s:
                continue
            v = victim_val + ni(-jitter, jitter)
            out.append((s, e, max(1, v)))
            cur = e + ni(0, 2)
    return out


def _noise(ni, T, n, val_lo, val_hi, len_lo, len_hi):
    out = []
    for _ in range(n):
        length = ni(len_lo, len_hi)
        s = ni(0, max(0, T - length - 1))
        e = s + length
        v = ni(val_lo, val_hi)
        out.append((s, e, v))
    return out


def _build_instance(spec):
    """spec: (seed, T, R, trap_windows, n_noise, val_lo, val_hi, len_lo, len_hi)."""
    seed, T, R, traps, n_noise, val_lo, val_hi, len_lo, len_hi = spec
    ni = _rng(seed)
    stays = []
    for (start, length, K, bv, vv, jitter) in traps:
        stays.extend(_trap_window(ni, start, length, R, K, bv, vv, jitter))
    stays.extend(_noise(ni, T, n_noise, val_lo, val_hi, len_lo, len_hi))
    N = len(stays)

    # clash roster from TRUE overlap
    edges = []
    for i in range(N):
        si, ei, _ = stays[i]
        for j in range(i + 1, N):
            sj, ej, _ = stays[j]
            if si < ej and sj < ei:
                edges.append((i, j))

    # scramble booking numbers so calendar order is not visible in the id order
    perm = list(range(N))
    for i in range(N - 1, 0, -1):
        j = ni(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    newpos = [0] * N
    for old, new in enumerate(perm):
        newpos[old] = new
    pub_stays = [None] * N
    for old in range(N):
        pub_stays[newpos[old]] = stays[old]
    pub_edges = sorted((min(newpos[a], newpos[b]), max(newpos[a], newpos[b])) for a, b in edges)

    return {
        "name": f"inst{seed}",
        "n": N,
        "rooms": R,
        "value": [v for (_, _, v) in pub_stays],
        "edges": pub_edges,
        "_start": [s for (s, _, _) in pub_stays],
        "_end": [e for (_, e, _) in pub_stays],
    }


def _build_instances():
    specs = [
        # ---- plain random, sanity ----
        (101, 40, 4, [], 16, 5, 40, 3, 14, "random-small"),
        (102, 50, 4, [], 24, 5, 45, 3, 16, "random-medium"),
        (103, 45, 3, [], 30, 5, 45, 3, 12, "random-dense"),
        # ---- TRAP: one blocker window each, R rooms, K victim tracks ----
        (201, 40, 2, [(2, 34, 4, 60, 20, 4)], 4, 3, 15, 2, 5, "trap-r2"),
        (202, 60, 3, [(4, 50, 4, 65, 22, 4)], 6, 3, 15, 2, 5, "trap-r3"),
        # ---- plain random, held out ----
        (301, 70, 4, [], 34, 5, 45, 3, 18, "random-large"),
        # ---- TRAP: heavier, more tracks ----
        (401, 55, 3, [(3, 48, 5, 55, 20, 3)], 8, 3, 12, 2, 5, "trap-heavy"),
        # ---- plain random, larger held out ----
        (501, 90, 5, [], 45, 5, 50, 3, 20, "random-xl"),
        (502, 100, 5, [], 55, 5, 50, 3, 22, "random-xl-dense"),
        # ---- TRAP: multiple trap windows plus background noise, largest ----
        (601, 80, 4, [(2, 30, 4, 60, 20, 4), (42, 34, 4, 55, 18, 4)], 18, 5, 30, 3, 12, "trap-multi"),
    ]
    out = []
    for seed, T, R, traps, n_noise, val_lo, val_hi, len_lo, len_hi, tag in specs:
        inst = _build_instance((seed, T, R, traps, n_noise, val_lo, val_hi, len_lo, len_hi))
        inst["tag"] = tag
        out.append(inst)
    return out


# ----------------------------- references (hidden data) --------------------
def _adj(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _cost_given_order(n, adj, values, R):
    """WEAK reference clerk: seat in (already scrambled) id order, first-fit
    into rooms 0..R-1 purely from the public clash graph."""
    room_members = [set() for _ in range(R)]
    spill = 0
    for i in range(n):
        placed = False
        for r in range(R):
            if not (adj[i] & room_members[r]):
                room_members[r].add(i)
                placed = True
                break
        if not placed:
            spill += values[i]
    return spill


def _cost_lb(n, starts, ends, values, R):
    """Loose, generally-unreachable per-day lower bound on spill."""
    lb = 0
    for t in sorted(set(starts)):
        active = [i for i in range(n) if starts[i] <= t < ends[i]]
        c = len(active)
        if c > R:
            excess = c - R
            vs = sorted(values[i] for i in active)
            lb = max(lb, sum(vs[:excess]))
    return lb


def _score(inst, answer):
    n = inst["n"]
    R = inst["rooms"]
    values = inst["value"]
    edges = inst["edges"]
    if not isinstance(answer, dict):
        return False, None
    room = answer.get("room")
    if not isinstance(room, list) or len(room) != n:
        return False, None
    for r in room:
        if isinstance(r, bool) or not isinstance(r, int):
            return False, None
        if r < -1 or r >= R:
            return False, None
    members = [[] for _ in range(R)]
    for i, r in enumerate(room):
        if r >= 0:
            members[r].append(i)
    adj = _adj(n, edges)
    for r in range(R):
        grp = members[r]
        for a in range(len(grp)):
            ia = grp[a]
            for b in range(a + 1, len(grp)):
                if grp[b] in adj[ia]:
                    return False, None
    spill = sum(values[i] for i, r in enumerate(room) if r == -1)
    return True, spill


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        n = inst["n"]
        R = inst["rooms"]
        values = inst["value"]
        edges = inst["edges"]
        starts = inst["_start"]
        ends = inst["_end"]
        adj = _adj(n, edges)
        base = _cost_given_order(n, adj, values, R)
        lb = _cost_lb(n, starts, ends, values, R)
        denom = base - lb
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "n": n, "rooms": R,
                  "edges": [list(e) for e in edges], "value": list(values)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, spill = _score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (base - spill) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
