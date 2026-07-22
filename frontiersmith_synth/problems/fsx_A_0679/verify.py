import sys
import math


def fail(msg):
    print("INVALID: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def rank_mod_p(rows, ncols, p):
    """Gaussian elimination rank over GF(p). rows: list of list[int]."""
    mat = [r[:] for r in rows]
    nrows = len(mat)
    rank = 0
    for col in range(ncols):
        piv = None
        for r in range(rank, nrows):
            if mat[r][col] % p != 0:
                piv = r
                break
        if piv is None:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        inv = pow(mat[rank][col], p - 2, p)
        mat[rank] = [(x * inv) % p for x in mat[rank]]
        for r in range(nrows):
            if r != rank and mat[r][col] % p != 0:
                f = mat[r][col]
                row_r = mat[r]
                row_p = mat[rank]
                mat[r] = [(row_r[k] - f * row_p[k]) % p for k in range(ncols)]
        rank += 1
        if rank == nrows:
            break
    return rank


def build_baseline(R, C, p):
    """A safe, always-fully-recoverable, low-raw-count reference construction:
    raw cells = a minimal covering of every row and every column (size <= R+C),
    everything else is parity with a globally-injective Cauchy-style tag so it
    is guaranteed full rank (way over-provisioned parity)."""
    raw_cells = set()
    for i in range(R):
        raw_cells.add((i, i % C))
    for j in range(C):
        raw_cells.add((j % R, j))
    data_idx = {}
    cnt = 0
    for i in range(R):
        for j in range(C):
            if (i, j) in raw_cells:
                data_idx[(i, j)] = cnt
                cnt += 1
    parity_coef = {}
    for i in range(R):
        for j in range(C):
            if (i, j) in raw_cells:
                continue
            x = (i * C + j + R * C + 1000) % p
            d = {}
            for (ii, jj), y in data_idx.items():
                denom = (x - y) % p
                if denom == 0:
                    denom = 1
                d[y] = pow(denom, p - 2, p)
            parity_coef[(i, j)] = d
    is_raw = [[(i, j) in raw_cells for j in range(C)] for i in range(R)]
    F, _ = score_construction(R, C, p, is_raw, parity_coef, cnt)
    return F


def score_construction(R, C, p, is_raw, parity_coef, raw_count):
    data_idx = {}
    cnt = 0
    for i in range(R):
        for j in range(C):
            if is_raw[i][j]:
                data_idx[(i, j)] = cnt
                cnt += 1
    total_frac = 0.0
    for r in range(R):
        for c in range(C):
            erased = set()
            for i in range(R):
                erased.add((i, c))
            for j in range(C):
                erased.add((r, j))
            erased_data = [data_idx[(i, j)] for (i, j) in erased if is_raw[i][j]]
            if not erased_data:
                total_frac += 1.0
                continue
            elist = sorted(set(erased_data))
            eidx = {v: k for k, v in enumerate(elist)}
            rows = []
            for i in range(R):
                for j in range(C):
                    if (not is_raw[i][j]) and (i, j) not in erased:
                        coefd = parity_coef[(i, j)]
                        vec = [0] * len(elist)
                        any_nonzero = False
                        for idx, co in coefd.items():
                            if idx in eidx:
                                v = co % p
                                if v != 0:
                                    vec[eidx[idx]] = v
                                    any_nonzero = True
                        if any_nonzero:
                            rows.append(vec)
            rk = rank_mod_p(rows, len(elist), p)
            total_frac += rk / len(elist)
    avg_frac = total_frac / (R * C)
    F = raw_count * avg_frac
    return F, avg_frac


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> [<ans>]")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    if len(toks) < 3:
        fail("bad input file")
    R, C, p = int(toks[0]), int(toks[1]), int(toks[2])
    N = R * C

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception:
        fail("cannot read output")

    # Bounded, strict token-stream parse. Reject non-finite / garbage / extra
    # tokens by construction: every token must parse as a plain base-10 int
    # via int(); int() rejects 'nan', 'inf', floats, etc.
    pos = 0
    ntok = len(out_toks)

    def next_tok():
        nonlocal pos
        if pos >= ntok:
            fail("output truncated")
        t = out_toks[pos]
        pos += 1
        return t

    def next_int():
        t = next_tok()
        try:
            v = int(t)
        except Exception:
            fail("non-integer token %r" % t)
        if not math.isfinite(v):
            fail("non-finite token %r" % t)
        return v

    # First pass: read the R*C cell tags to find how many D cells there are
    # and to know, for parity lines, how many (idx,coef) pairs to consume.
    # We must parse in one pass because P-lines have variable length; a cell's
    # tag ('D' or 'P') is itself a token, not int-parseable for 'D', so peek it.
    cell_kind = []  # 'D' or list of (idx, coef) raw for P
    raw_count_running = 0
    for cellno in range(N):
        if pos >= ntok:
            fail("output truncated (expected %d cell records)" % N)
        tag = out_toks[pos]
        pos += 1
        if tag == "D":
            cell_kind.append(("D", None))
            raw_count_running += 1
        elif tag == "P":
            k = next_int()
            if k < 1 or k > N:
                fail("bad k=%d for parity cell" % k)
            pairs = []
            for _ in range(k):
                idx = next_int()
                co = next_int()
                if co < 0 or co >= p:
                    fail("coefficient out of range")
                pairs.append((idx, co))
            seen_idx = set(idx for idx, _ in pairs)
            if len(seen_idx) != len(pairs):
                fail("duplicate raw index inside one P line")
            cell_kind.append(("P", pairs))
        else:
            fail("bad cell tag %r (want D or P)" % tag)

    if pos != ntok:
        fail("trailing garbage after %d cell records" % N)

    raw_count = raw_count_running
    if raw_count < 1:
        fail("no raw (D) cells at all")

    # assign data indices in row-major discovery order, validate P index ranges
    is_raw = [[False] * C for _ in range(R)]
    data_idx = {}
    k = 0
    cellno = 0
    for i in range(R):
        for j in range(C):
            kind, _ = cell_kind[cellno]
            cellno += 1
            if kind == "D":
                is_raw[i][j] = True
                data_idx[(i, j)] = k
                k += 1

    parity_coef = {}
    cellno = 0
    for i in range(R):
        for j in range(C):
            kind, payload = cell_kind[cellno]
            cellno += 1
            if kind == "P":
                d = {}
                for idx, co in payload:
                    if idx < 0 or idx >= raw_count:
                        fail("parity index %d out of range (raw_count=%d)" % (idx, raw_count))
                    d[idx] = (d.get(idx, 0) + co) % p
                parity_coef[(i, j)] = d

    F, avg_frac = score_construction(R, C, p, is_raw, parity_coef, raw_count)
    B = build_baseline(R, C, p)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("raw=%d avg_frac=%.6f F=%.6f baseline=%.6f" % (raw_count, avg_frac, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
