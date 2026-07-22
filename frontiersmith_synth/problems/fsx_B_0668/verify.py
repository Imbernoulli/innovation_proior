#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- replays the sequential second-price-with-reserve
estate auction in the participant's chosen order/reserves and prints Ratio: <f>."""
import sys

RESERVE_MAX = 5_000_000
TOKEN_LEN_CAP = 15


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def safe_int(tok):
    if not tok or len(tok) > TOKEN_LEN_CAP:
        return None
    s = tok
    if s[0] in "+-":
        s2 = s[1:]
    else:
        s2 = s
    if not s2.isdigit():
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    values = []
    for _ in range(n):
        values.append([int(next(it)) for _ in range(m)])
    budgets = [int(next(it)) for _ in range(m)]
    return n, m, values, budgets


def simulate(n, m, values, budgets, pairs):
    """pairs: list of (lot0_index, reserve) in sale order. Returns total revenue."""
    remaining = budgets[:]
    revenue = 0
    for lot0, r in pairs:
        row = values[lot0]
        best_bid = -1
        best_j = -1
        second_bid = -1
        for j in range(m):
            v = row[j]
            if v <= 0:
                continue
            rem = remaining[j]
            b = v if v < rem else rem
            if b < r:
                continue
            if b > best_bid:
                second_bid = best_bid
                best_bid = b
                best_j = j
            elif b > second_bid:
                second_bid = b
        if best_j < 0:
            continue
        price = r if second_bid < 0 else (second_bid if second_bid > r else r)
        revenue += price
        remaining[best_j] -= price
    return revenue


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m, values, budgets = read_instance(in_path)

    with open(out_path) as f:
        raw_toks = f.read().split()

    if len(raw_toks) != 2 * n:
        fail(f"expected {2*n} tokens, got {len(raw_toks)}")

    parsed = []
    for t in raw_toks:
        v = safe_int(t)
        if v is None:
            fail(f"malformed token {t!r}")
        parsed.append(v)

    item_ids = parsed[0::2]
    reserves = parsed[1::2]

    seen = [False] * (n + 1)
    pairs = []
    for idx in range(n):
        item = item_ids[idx]
        r = reserves[idx]
        if item < 1 or item > n:
            fail(f"item id {item} out of range")
        if seen[item]:
            fail(f"item id {item} repeated")
        seen[item] = True
        if r < 0 or r > RESERVE_MAX:
            fail(f"reserve {r} out of range")
        pairs.append((item - 1, r))

    F = simulate(n, m, values, budgets, pairs)

    base_pairs = [(i, 0) for i in range(n)]
    B = simulate(n, m, values, budgets, base_pairs)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F} B={B}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
