#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for the cache-cooperative
tiling problem.

The participant's artifact is a tiling+padding+inner-loop-order PLAN for the
canonical triple-nested GEMM  C[i][j] += A[i][k]*B[k][j]  over i,j,k in [0,N).
The OUTER block-visitation order is fixed (I,K,J major) -- the plan controls
tile SIZE (the loop-nest-tiling mechanism), per-array row PADDING (the
conflict-miss-padding mechanism) and the INNER loop order within a tile (the
prefetch-friendliness mechanism). Tile sizes are structurally capped at
N//3, so every legal plan visits >= 3 blocks per axis: it is impossible to
"collapse" the schema into a single global element order and skip real
tiling. Because the fixed schema always visits every (i,j,k) triple exactly
once (block ranges are generated with range(0,N,T), clipped to N), the sum
C=A*B is automatically exact regardless of tile size/padding/order (addition
is associative) -- the real "equivalence" gate here is the SCHEMA gate:
every field must be a well-formed, in-range token, or the plan is rejected
outright (Ratio: 0.0).

Given a feasible plan we replay the EXACT memory-address trace it induces
(word addresses, row-major layout with per-array padding) through a
deterministic set-associative LRU cache simulator (with a simple per-stream
sequential prefetcher) and count total cache misses over the 3*N^3 memory
events. Fewer misses -> higher score. The internal baseline B is the
simplest LEGAL plan: the SMALLEST allowed tile (T=min(2,N//3)), zero
padding, canonical i,j,k inner order -- i.e. tiling with no capacity/order/
padding reasoning behind it at all.
"""
import sys

OUTER_ORDER = "IKJ"   # fixed block-visitation order (not part of the artifact)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    C = int(next(it)); L = int(next(it)); A = int(next(it))
    PAD_MAX = int(next(it))
    return N, C, L, A, PAD_MAX


def parse_plan(path):
    """Strict schema parse. Returns dict or None (+ reason) on ANY malformity."""
    try:
        raw = open(path).read()
    except Exception:
        return None, "cannot read output"
    toks = raw.split()
    if len(toks) != 7:
        return None, f"expected exactly 7 tokens, got {len(toks)}"
    int_toks = toks[:6]
    inner_order = toks[6]
    parsed = []
    for t in int_toks:
        # reject anything that isn't a plain (optionally signed) decimal integer
        # literal -- blocks "nan"/"inf"/"1e3"/"3.5" etc. outright.
        core = t[1:] if t[:1] in "+-" else t
        if core == "" or not core.isdigit():
            return None, f"non-integer token {t!r}"
        parsed.append(int(t))
    Ti, Tj, Tk, padA, padB, padC = parsed
    if sorted(inner_order) != ["i", "j", "k"]:
        return None, f"inner_order {inner_order!r} is not a permutation of ijk"
    return {
        "Ti": Ti, "Tj": Tj, "Tk": Tk,
        "padA": padA, "padB": padB, "padC": padC,
        "outer": OUTER_ORDER, "inner": inner_order,
    }, None


def tile_bounds(N):
    """Every legal tile size must produce >= 3 blocks (max_t) AND actually span more
    than one element (min_t) -- min_t=1 would let a "tile" degenerate into a single
    scalar element, silently turning the fixed block-visitation order into a free
    global element order and bypassing tiling/padding entirely."""
    max_t = max(1, N // 3)
    min_t = min(2, max_t)
    return min_t, max_t


def validate_plan(plan, N, PAD_MAX):
    min_t, max_t = tile_bounds(N)
    for key in ("Ti", "Tj", "Tk"):
        v = plan[key]
        if not (min_t <= v <= max_t):
            return f"{key}={v} out of range [{min_t},{max_t}] (must genuinely tile)"
    for key in ("padA", "padB", "padC"):
        v = plan[key]
        if not (0 <= v <= PAD_MAX):
            return f"{key}={v} out of range [0,{PAD_MAX}]"
    return None


def blocks(T, n):
    out = []
    b = 0
    while b < n:
        out.append((b, min(b + T, n)))
        b += T
    return out


def simulate_misses(N, C, L, A, plan):
    S = C // (L * A)
    Ti, Tj, Tk = plan["Ti"], plan["Tj"], plan["Tk"]
    padA, padB, padC = plan["padA"], plan["padB"], plan["padC"]
    outer_order, inner_order = plan["outer"], plan["inner"]

    rowA, rowB, rowC = N + padA, N + padB, N + padC
    baseA = 0
    baseB = baseA + N * rowA
    baseC = baseB + N * rowB

    # cache: S sets, each a small MRU-ordered list of resident line ids (len<=A)
    sets = [[] for _ in range(S)]
    misses = 0
    last_line = {"A": None, "B": None, "C": None}

    def access(line):
        s = line % S
        bucket = sets[s]
        try:
            bucket.remove(line)
            bucket.insert(0, line)
            return True
        except ValueError:
            bucket.insert(0, line)
            if len(bucket) > A:
                bucket.pop()
            return False

    def prefetch_insert(line):
        s = line % S
        bucket = sets[s]
        if line not in bucket:
            bucket.insert(0, line)
            if len(bucket) > A:
                bucket.pop()

    def do_access(addr, stream):
        nonlocal misses
        line = addr // L
        if not access(line):
            misses += 1
        ll = last_line[stream]
        if ll is not None and line == ll + 1:
            prefetch_insert(line + 1)   # free (unaccounted) sequential prefetch
        last_line[stream] = line

    dim_ranges = {
        "I": blocks(Ti, N), "J": blocks(Tj, N), "K": blocks(Tk, N),
    }
    outer_lists = [dim_ranges[d] for d in outer_order]

    def rec_outer(pos, block):
        if pos == 3:
            ii0, ii1 = block["I"]; jj0, jj1 = block["J"]; kk0, kk1 = block["K"]
            inner_iters = []
            for d in inner_order:
                if d == "i":
                    inner_iters.append(range(ii0, ii1))
                elif d == "j":
                    inner_iters.append(range(jj0, jj1))
                else:
                    inner_iters.append(range(kk0, kk1))
            a_it, b_it, c_it = inner_iters
            for x0 in a_it:
                for x1 in b_it:
                    for x2 in c_it:
                        idx = {}
                        idx[inner_order[0]] = x0
                        idx[inner_order[1]] = x1
                        idx[inner_order[2]] = x2
                        i, j, k = idx["i"], idx["j"], idx["k"]
                        do_access(baseA + i * rowA + k, "A")
                        do_access(baseB + k * rowB + j, "B")
                        do_access(baseC + i * rowC + j, "C")
            return
        d = outer_order[pos]
        for lo, hi in dim_ranges[d]:
            block[d] = (lo, hi)
            rec_outer(pos + 1, block)

    rec_outer(0, {})
    return misses


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, C, L, A, PAD_MAX = read_instance(in_path)

    plan, err = parse_plan(out_path)
    if plan is None:
        print(f"Ratio: 0.0 (parse error: {err})")
        return 0
    err = validate_plan(plan, N, PAD_MAX)
    if err is not None:
        print(f"Ratio: 0.0 (infeasible: {err})")
        return 0

    F = simulate_misses(N, C, L, A, plan)

    min_t, _ = tile_bounds(N)
    baseline_plan = {
        "Ti": min_t, "Tj": min_t, "Tk": min_t,
        "padA": 0, "padB": 0, "padC": 0,
        "outer": OUTER_ORDER, "inner": "ijk",
    }
    B = simulate_misses(N, C, L, A, baseline_plan)

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print(f"misses={F} baseline={B}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
