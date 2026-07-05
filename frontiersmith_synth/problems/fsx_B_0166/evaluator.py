#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0166 -- "Glacier Sensor Net: Predicate Prefix-Cache Ordering"
(family: offline-decision-policy; format B, quality-metric).

THEME.  A remote glacier sensor network answers streams of monitoring queries.  Every
query is a conjunction of predicates over sensor CHANNELS (ice temperature, tilt, GPS
drift, meltwater conductivity, ...).  A query engine evaluates the predicates of a query
one channel at a time and MEMOIZES every partial-evaluation *prefix* it has ever produced
(a prefix KV-cache): the first time a channel-sequence prefix is computed it is a cache
MISS (real work over the raw sensor logs); any later query whose leading channels reproduce
that same prefix gets a cache HIT for free.

The catch: within a single query the predicates commute, so the engine may evaluate the
channels of a query in ANY order it likes -- but it must fix ONE GLOBAL channel ordering
and canonicalize every query's predicates into that order before caching.  Choosing a good
global order clusters channels that are frequently queried together into shared leading
prefixes, so many queries collapse onto the same cached path and the total compute (number
of distinct prefixes ever materialized = trie nodes) drops sharply.  A poor order scatters
related channels and forces the engine to recompute almost everything.

This is the SQL predicate-column reordering / prefix KV-cache hit-rate problem skinned as an
offline sensor-net policy: pick a permutation of channels to MINIMIZE the number of distinct
cached prefixes (equivalently MAXIMIZE the prefix cache hit rate) on the query stream.

OFFLINE DECISION.  The candidate sees only a TRAINING stream of queries and must commit to a
single global channel order.  It is scored on a DISJOINT held-out stream drawn from the same
sensor-net query distribution -- so the order must GENERALIZE, not memorize the training log.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "N": int,                     # number of sensor channels, ids 0..N-1
             "train_queries": [[c,...],..]} # each query = sorted list of DISTINCT channel ids
  stdout: ONE JSON object:
            {"order": [p_0, ..., p_{N-1}]}  # a PERMUTATION of 0..N-1 (the global channel order)

  VALID iff `order` is a list of exactly N integers that is a permutation of 0..N-1.
  Anything else (wrong length, repeats, out-of-range, non-int, crash, timeout, non-JSON) -> 0.0.

SCORING (deterministic; no wall-time).  Per instance, on the HELD-OUT query stream, we count
distinct materialized prefixes (trie nodes) under three orders:
    q_base = trie nodes under the natural channel order 0,1,...,N-1  (weak baseline)
    q_cand = trie nodes under the candidate's order
    q_ref  = trie nodes under an internal near-optimal order found by seeded local search
             on the HELD-OUT stream (a strong, generally-unreachable reference)
  and normalize with an affine anchor (natural order -> 0.1, near-optimal ref -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_ref), 0, 1 )
  Matching the natural order scores ~0.1; reaching the local-search reference scores 1.0;
  doing worse than the natural order scores < 0.1.  Because the reference is computed on the
  held-out stream the candidate never sees, even good generalizing orders stay strictly below
  1.0 on most instances -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC (training) instance.  The held-out
stream and the reference order are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, random
import isorun


# ----------------------------- instance family -----------------------------
def _build_instance(seed, N, K, motif_lo, motif_hi, n_train, n_test):
    """Deterministic sensor-net query streams built from co-occurring channel motifs
    (sensor 'stations' that tend to be queried together)."""
    rng = random.Random(seed)
    cols = list(range(N))
    motifs = [rng.sample(cols, rng.randint(motif_lo, motif_hi)) for _ in range(K)]
    pop = [rng.randint(1, 5) for _ in range(K)]

    def gen_query():
        m1 = rng.choices(range(K), weights=pop, k=1)[0]
        s = set(motifs[m1])
        if rng.random() < 0.45:                       # sometimes two stations at once
            m2 = rng.choices(range(K), weights=pop, k=1)[0]
            s |= set(motifs[m2])
        if rng.random() < 0.15:                       # occasional ad-hoc channel
            s.add(rng.randrange(N))
        return sorted(s)

    train = [gen_query() for _ in range(n_train)]
    test = [gen_query() for _ in range(n_test)]
    return {"name": f"glacier{seed}", "N": N,
            "train_queries": train, "test_queries": test}


def _build_instances():
    """Deterministic instance family. (seed, N, K, motif_lo, motif_hi, n_train, n_test)."""
    specs = [
        (101, 16, 7, 3, 5, 45, 40),
        (102, 18, 8, 3, 5, 45, 40),
        (103, 18, 8, 3, 5, 45, 40),
        (104, 20, 9, 3, 5, 45, 40),
        (105, 16, 6, 3, 5, 45, 40),
        (106, 18, 7, 3, 5, 45, 40),
        (107, 20, 8, 3, 5, 45, 40),
        (108, 22, 9, 3, 5, 45, 40),
        # harder / held-out generalization: less training data, larger test streams
        (211, 20, 7, 3, 5, 30, 55),
        (212, 22, 8, 3, 6, 30, 55),
        (213, 18, 6, 3, 5, 35, 50),
        (214, 24, 10, 3, 6, 35, 55),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- trie cost -----------------------------------
def _nodes(queries, rank):
    """Number of distinct materialized prefixes (trie nodes) when each query's
    channels are evaluated in the order given by `rank` (channel -> position)."""
    trie = {}
    cnt = 0
    for q in queries:
        seq = sorted(q, key=lambda c: rank[c])
        node = trie
        for c in seq:
            nxt = node.get(c)
            if nxt is None:
                nxt = {}
                node[c] = nxt
                cnt += 1
            node = nxt
    return cnt


def _rank_of(order):
    return {c: i for i, c in enumerate(order)}


# ----------------------------- reference (near-optimal on held-out) --------
def _freq_order(N, queries):
    freq = [0] * N
    for q in queries:
        for c in q:
            freq[c] += 1
    return sorted(range(N), key=lambda c: (-freq[c], c))


def _cooc_order(N, queries):
    C = [[0] * N for _ in range(N)]
    freq = [0] * N
    for q in queries:
        for c in q:
            freq[c] += 1
        L = len(q)
        for a in range(L):
            qa = q[a]
            for b in range(a + 1, L):
                qb = q[b]
                C[qa][qb] += 1
                C[qb][qa] += 1
    start = max(range(N), key=lambda c: (freq[c], c))
    order = [start]
    placed = {start}
    while len(order) < N:
        best = None
        best_key = None
        for c in range(N):
            if c in placed:
                continue
            sc = sum(C[c][p] for p in order)
            key = (sc, freq[c], -c)
            if best_key is None or key > best_key:
                best_key = key
                best = c
        order.append(best)
        placed.add(best)
    return order


def _hill_climb(N, queries, start, passes=8):
    order = list(start)
    best = _nodes(queries, _rank_of(order))
    improved = True
    p = 0
    while improved and p < passes:
        improved = False
        p += 1
        for i in range(N):
            for j in range(i + 1, N):
                order[i], order[j] = order[j], order[i]
                c = _nodes(queries, _rank_of(order))
                if c < best:
                    best = c
                    improved = True
                else:
                    order[i], order[j] = order[j], order[i]
    return order, best


def _reference_nodes(N, test):
    """Strong, generally-unreachable reference: seeded multi-start local search on the
    HELD-OUT stream. The candidate never sees this stream, so it keeps headroom."""
    starts = [_freq_order(N, test), _cooc_order(N, test)]
    rng = random.Random(917 + N)
    for _ in range(4):
        s = list(range(N))
        rng.shuffle(s)
        starts.append(s)
    best = None
    for st in starts:
        _, c = _hill_climb(N, test, st, passes=8)
        if best is None or c < best:
            best = c
    return best


# ----------------------------- validation ----------------------------------
def _cand_nodes(inst, answer):
    """Validate the candidate order against the instance; return held-out trie-node
    count, or None if the answer is malformed."""
    if not isinstance(answer, dict):
        return None
    order = answer.get("order")
    if not isinstance(order, list):
        return None
    N = inst["N"]
    if len(order) != N:
        return None
    seen = [False] * N
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= N or seen[v]:
            return None
        seen[v] = True
    return _nodes(inst["test_queries"], _rank_of(order))


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        N = inst["N"]
        test = inst["test_queries"]
        q_base = _nodes(test, _rank_of(list(range(N))))
        q_ref = _reference_nodes(N, test)
        denom = q_base - q_ref
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "N": N,
                  "train_queries": [list(q) for q in inst["train_queries"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _cand_nodes(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (q_base - q_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
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
