#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0880 -- "Container Yard: Corridor Reservation Dispatch"
(family: spacetime-reservation-dispatch; format B, quality-metric).

THEME.  A container-terminal yard is modeled as a ring of R junctions (the yard's
perimeter travel lane, wide enough for any number of AGVs to pass each other freely)
plus a handful of SINGLE-LANE SHORTCUT aisles that cut straight across the yard
between two junctions (a gap between container-stack blocks just wide enough for
one vehicle). AGVs must ferry a sequence of container-move requests, each a
(source junction, destination junction, earliest-release time). Every edge has an
integer traversal length (time units); a shortcut is much shorter than walking
around the ring, so it is tempting -- but at most one AGV may physically occupy a
shortcut's aisle at a time.  If two AGVs' occupancy WINDOWS on the same shortcut
overlap while traveling in OPPOSITE directions, they meet head-on: this is a
DEADLOCK.  The aisle then JAMS from that moment on -- every occupancy window on
that shortcut (from ANY move, including ones dispatched later) that starts at or
after the jam instant fails, so one head-on collision can strand many later moves,
not just the two AGVs directly involved (throughput-blocking).  Overlapping
windows in the SAME direction are also physically impossible (the lane fits one
vehicle) and both those moves fail too, but they do not jam the aisle for others.

The operator MAXIMIZES completed throughput: each move that reaches its
destination without violating the single-lane constraint contributes a score in
[0,1] -- 1.0 if it arrives at the graph-shortest-path-ignoring-traffic time, decayed
mildly (rate BETA) for every time unit later it finishes (from waiting or detouring).
Failed / not-attempted moves contribute 0.  The tension: racing every AGV down its
shortest path (through the shortcuts, departing immediately) risks a head-on that
wipes out a chunk of the fleet's throughput; reserving each shortcut's time-windows
in advance -- delaying entry, or occasionally taking the longer ring-only detour --
trades a little time for guaranteed passage.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n_nodes": int,                 # ring junctions 0..n_nodes-1
             "edges": [{"id":int,"u":int,"v":int,"length":int,"shared":bool}, ...],
             "moves": [{"id":int,"src":int,"dst":int,"release":number}, ...],
             "horizon": number}
          Every edge is traversable in either direction. "shared" edges are the
          single-lane shortcuts described above; all other edges (the ring) have
          unlimited capacity -- any number of AGVs may use them simultaneously.
  stdout: ONE JSON object:
            {"moves": [{"id":int, "path":[n0,n1,...,nk], "times":[t0,t1,...,tk]}, ...]}
          `path` must start at that move's src and end at its dst, and every
          consecutive pair must be a real edge. `times` gives the time the AGV is
          AT each node of `path` (departure/arrival instants): t0 >= release, and
          for every edge (path[i],path[i+1]): times[i+1]-times[i] >= edge length
          (an AGV may wait at a node before continuing, never go faster than an
          edge's length). A move whose id is omitted from the list is treated as
          not attempted (0 credit). A move with any other malformed / out-of-range
          / non-finite / disconnected entry is invalid and scores 0 for that move
          alone (it does not disqualify the rest of the candidate's answer).
          A finish time beyond `horizon`, a crash, a timeout, or non-JSON output
          scores that whole instance 0.0.

SCORING (deterministic; no wall-time).  For every move whose (path,times) are
structurally valid and within the horizon, we record its shared-edge occupancy
windows [times[i], times[i+1]) with a direction. For each shared edge, every pair
of overlapping windows fails BOTH of those moves; if the pair travels in opposite
directions this is additionally a DEADLOCK -- from the later of the two start
times onward, every window on that edge (any move) fails too (cascading block).
A move that survives contributes  1 - BETA*max(0, finish-ideal_finish)/ideal_finish
(clipped to [0,1]), where ideal_finish = release + shortest-path time ignoring all
traffic (BETA = 0.35); a failed / unattempted move contributes 0.  Summing over
all moves in the instance gives raw_cand.  We also compute raw_base -- the SAME
formula applied to the naive "everyone takes the graph-shortest path and departs
immediately at release" placement (the recipe an uncoordinated dispatcher writes
first; on trap instances it deadlocks itself). Final per-instance score:
    r = clamp( 0.1 + 0.9 * (raw_cand - raw_base) / (UB_MULT*M - raw_base), 0, 1 )
  where M = number of moves and UB_MULT = 1.15 (a loose, generally unreachable
  upper bound so even an excellent dispatcher leaves headroom below 1.0).
  Reproducing the naive placement scores ~0.1; doing strictly worse scores < 0.1
  (down to 0); resolving conflicts that the naive dispatcher deadlocked on scores
  much higher.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. All references
(naive baseline, ideal shortest times) and all validation happen in THIS parent
process, so a frame-walking / introspecting candidate learns nothing extra.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, heapq
import isorun

BETA = 0.35
UB_MULT = 1.15


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance construction ------------------------
def _build_instance(seed, R, L, shortcut_defs, filler_count):
    rng = _rng(seed)
    edges = []
    eid = 0
    for i in range(R):
        edges.append({"id": eid, "u": i, "v": (i + 1) % R, "length": L, "shared": False})
        eid += 1
    shortcuts = []
    for sdef in shortcut_defs:
        a = sdef["a"]; half = sdef["half"]; Ls = sdef["len"]
        b = (a + half) % R
        edges.append({"id": eid, "u": a, "v": b, "length": Ls, "shared": True})
        shortcuts.append({"eid": eid, "a": a, "b": b, "len": Ls, "trap": sdef["trap"]})
        eid += 1

    moves = []
    mid = 0
    base_time = R * L  # slack so every subtraction below stays nonnegative

    for sc in shortcuts:
        # src1/dst1 sit near a / near b on the SAME long ring arc (small offsets
        # {0,1}) so the only competitive route between them is via the shortcut.
        off1 = rng(0, 1); off2 = rng(0, 1)
        src1 = (sc["a"] - off1) % R
        dst1 = (sc["b"] + off2) % R
        t1 = base_time + rng(0, 5)
        moves.append({"id": mid, "src": src1, "dst": dst1, "release": t1}); mid += 1
        entry1 = t1 + off1 * L

        # src2/dst2 mirror the reverse direction (b -> a) on the same arc.
        off3 = rng(0, 1); off4 = rng(0, 1)
        src2 = (sc["b"] + off3) % R
        dst2 = (sc["a"] - off4) % R
        if sc["trap"]:
            # force move2 to ENTER the shortcut at the exact instant move1 does ->
            # guaranteed head-on under naive (no-wait) dispatch.
            release2 = entry1 - off3 * L
            if release2 < 0:
                release2 = 0
        else:
            # scheduled comfortably after move1 has fully cleared -> no conflict.
            release2 = entry1 + sc["len"] + rng(8, 14)
        moves.append({"id": mid, "src": src2, "dst": dst2, "release": release2}); mid += 1

        if sc["trap"]:
            # a later same-direction move that would succeed if the aisle weren't
            # jammed by the head-on above -- demonstrates throughput-blocking.
            off5 = rng(0, 1); off6 = rng(0, 1)
            src3 = (sc["a"] - off5) % R
            dst3 = (sc["b"] + off6) % R
            release3 = entry1 + sc["len"] + rng(2, 6)
            moves.append({"id": mid, "src": src3, "dst": dst3, "release": release3}); mid += 1

    for _ in range(filler_count):
        span = rng(2, 4)
        s0 = rng(0, R - 1)
        d0 = (s0 + span) % R
        rel = rng(0, base_time + 40)
        moves.append({"id": mid, "src": s0, "dst": d0, "release": rel}); mid += 1

    horizon = base_time + R * L + 80
    return {"edges": edges, "moves": moves, "n_nodes": R, "horizon": horizon}


def _build_instances():
    specs = [
        (1001, 12, 4, [{"a": 1, "half": 6, "len": 3, "trap": True}], 3),
        (1002, 12, 4, [{"a": 2, "half": 6, "len": 3, "trap": True}], 4),
        (1003, 12, 4, [{"a": 3, "half": 6, "len": 3, "trap": False}], 4),
        (1004, 14, 4, [{"a": 1, "half": 7, "len": 3, "trap": True}], 4),
        (1005, 14, 4, [{"a": 4, "half": 7, "len": 3, "trap": False}], 5),
        (1006, 16, 4, [{"a": 2, "half": 8, "len": 4, "trap": True}], 5),
        (1007, 16, 4, [{"a": 5, "half": 8, "len": 4, "trap": False}], 5),
        (1008, 20, 4, [{"a": 1, "half": 10, "len": 3, "trap": True},
                        {"a": 5, "half": 10, "len": 3, "trap": False}], 3),
        (1101, 18, 4, [{"a": 3, "half": 9, "len": 4, "trap": False}], 6),
        (1102, 18, 4, [{"a": 6, "half": 9, "len": 4, "trap": True}], 6),
        (1103, 20, 4, [{"a": 2, "half": 10, "len": 4, "trap": False}], 6),
        (1104, 20, 4, [{"a": 8, "half": 10, "len": 4, "trap": True}], 5),
    ]
    out = []
    for (seed, R, L, sdefs, filler) in specs:
        inst = _build_instance(seed, R, L, sdefs, filler)
        inst["name"] = f"yard{seed}"
        out.append(inst)
    return out


# ----------------------------- graph helpers ---------------------------------
def _adj(inst):
    n = inst["n_nodes"]
    adj = [[] for _ in range(n)]
    for e in inst["edges"]:
        adj[e["u"]].append((e["v"], e["length"]))
        adj[e["v"]].append((e["u"], e["length"]))
    return adj


def _dijkstra_dist(n, adj, src, dst):
    dist = [None] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] is not None and d > dist[u]:
            continue
        if u == dst:
            return d
        for v, w in adj[u]:
            nd = d + w
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist[dst]


def _edge_map(inst):
    return {frozenset((e["u"], e["v"])): e for e in inst["edges"]}


def _ideal_map(inst):
    adj = _adj(inst)
    n = inst["n_nodes"]
    return {mv["id"]: _dijkstra_dist(n, adj, mv["src"], mv["dst"]) for mv in inst["moves"]}


def _naive_answers(inst):
    """The 'obvious' recipe: graph-shortest path, depart at release, never wait."""
    n = inst["n_nodes"]
    adj = _adj(inst)
    emap = _edge_map(inst)
    answers = {}
    for mv in inst["moves"]:
        dist = [None] * n; prev = [None] * n
        dist[mv["src"]] = 0
        pq = [(0, mv["src"])]
        while pq:
            d, u = heapq.heappop(pq)
            if dist[u] is not None and d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(pq, (nd, v))
        path = [mv["dst"]]
        while path[-1] != mv["src"]:
            path.append(prev[path[-1]])
        path.reverse()
        t = mv["release"]; times = [t]
        for i in range(len(path) - 1):
            e = emap[frozenset((path[i], path[i + 1]))]
            t += e["length"]; times.append(t)
        answers[mv["id"]] = {"path": path, "times": times}
    return answers


# ----------------------------- validation + simulation ------------------------
def _valid_move(mv, path, times, emap, n):
    if not isinstance(path, list) or not isinstance(times, list):
        return False
    if len(path) < 2 or len(path) != len(times) or len(path) > 4 * n + 4:
        return False
    if path[0] != mv["src"] or path[-1] != mv["dst"]:
        return False
    for x in path:
        if isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= n:
            return False
    prev_t = None
    for t in times:
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            return False
        if t != t or t in (float("inf"), float("-inf")):
            return False
        if prev_t is not None and t < prev_t - 1e-9:
            return False
        prev_t = t
    if times[0] < mv["release"] - 1e-9:
        return False
    for i in range(len(path) - 1):
        key = frozenset((path[i], path[i + 1]))
        if key not in emap:
            return False
        e = emap[key]
        if times[i + 1] - times[i] < e["length"] - 1e-9:
            return False
    return True


def _intervals(path, times, emap):
    out = []
    for i in range(len(path) - 1):
        e = emap[frozenset((path[i], path[i + 1]))]
        if e["shared"]:
            d = 1 if path[i] == e["u"] else -1
            out.append((e["id"], times[i], times[i + 1], d))
    return out


def _simulate(inst, answers):
    n = inst["n_nodes"]; emap = _edge_map(inst); horizon = inst["horizon"]
    valid = {}; finish = {}; ivs_by_move = {}
    for mv in inst["moves"]:
        mid = mv["id"]
        ans = answers.get(mid)
        if not isinstance(ans, dict):
            continue
        path = ans.get("path"); times = ans.get("times")
        if not _valid_move(mv, path, times, emap, n):
            continue
        if times[-1] > horizon + 1e-9:
            continue
        valid[mid] = True; finish[mid] = times[-1]
        ivs_by_move[mid] = _intervals(path, times, emap)
    failed = set()
    by_edge = {}
    for mid, ivs in ivs_by_move.items():
        for (eid, s, t, d) in ivs:
            by_edge.setdefault(eid, []).append((s, t, d, mid))
    for eid, ivs in by_edge.items():
        jam_time = None
        m_ = len(ivs)
        for i in range(m_):
            s1, t1, d1, m1 = ivs[i]
            for j in range(i + 1, m_):
                s2, t2, d2, m2 = ivs[j]
                if s1 < t2 - 1e-9 and s2 < t1 - 1e-9:
                    failed.add(m1); failed.add(m2)
                    if d1 != d2:
                        jt = max(s1, s2)
                        if jam_time is None or jt < jam_time:
                            jam_time = jt
        if jam_time is not None:
            for (s, t, d, m) in ivs:
                if s >= jam_time - 1e-9:
                    failed.add(m)
    completed = [mid for mid in valid if mid not in failed]
    return completed, finish


def _raw_score(inst, answers, ideal):
    completed, finish = _simulate(inst, answers)
    rel_of = {mv["id"]: mv["release"] for mv in inst["moves"]}
    raw = 0.0
    for mid in completed:
        ideal_finish = rel_of[mid] + ideal[mid]
        f = finish[mid]
        extra = max(0.0, f - ideal_finish)
        denom = max(ideal_finish, 1.0)
        contrib = 1.0 - BETA * (extra / denom)
        if contrib < 0.0:
            contrib = 0.0
        raw += contrib
    return raw


def _parse_candidate(raw, inst):
    if not isinstance(raw, dict):
        return {}
    mv_list = raw.get("moves")
    if not isinstance(mv_list, list):
        return {}
    valid_ids = {m["id"] for m in inst["moves"]}
    out = {}
    for item in mv_list:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if isinstance(mid, bool) or not isinstance(mid, int) or mid not in valid_ids or mid in out:
            continue
        out[mid] = {"path": item.get("path"), "times": item.get("times")}
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        ideal = _ideal_map(inst)
        raw_base = _raw_score(inst, _naive_answers(inst), ideal)
        M = len(inst["moves"])
        ub = UB_MULT * M
        public = {"name": inst["name"], "n_nodes": inst["n_nodes"],
                  "edges": [dict(e) for e in inst["edges"]],
                  "moves": [dict(m) for m in inst["moves"]],
                  "horizon": inst["horizon"]}
        ans_raw, st = isorun.run_candidate(cand, public, timeout=10)
        if st != "OK":
            vec.append(0.0); continue
        try:
            answers = _parse_candidate(ans_raw, inst)
            raw_cand = _raw_score(inst, answers, ideal)
        except Exception:
            vec.append(0.0); continue
        denom = max(1e-9, ub - raw_base)
        r = 0.1 + 0.9 * (raw_cand - raw_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0); continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
