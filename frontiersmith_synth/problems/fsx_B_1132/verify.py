#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the biogas
digester problem. Prints "... Ratio: <float>". ans is an unused
placeholder (checker problems ignore it)."""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import simcore as sc


def fail(msg):
    print("INFEASIBLE: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    try:
        with open(path, "r") as f:
            return f.read().split()
    except Exception as e:
        fail(f"cannot read {path}: {e}")


def parse_int(tok, ctx):
    try:
        if "." in tok or "e" in tok.lower():
            raise ValueError
        return int(tok)
    except Exception:
        fail(f"expected integer for {ctx}, got {tok!r}")


def parse_float_strict(tok, ctx):
    try:
        v = float(tok)
    except Exception:
        fail(f"expected number for {ctx}, got {tok!r}")
        return None
    if v != v or v in (float("inf"), float("-inf")):
        fail(f"non-finite value for {ctx}: {tok!r}")
    return v


def parse_instance(path):
    toks = read_tokens(path)
    pos = [0]

    def nxt():
        if pos[0] >= len(toks):
            fail("instance truncated")
        v = toks[pos[0]]
        pos[0] += 1
        return v

    T = int(nxt()); K = int(nxt())
    alpha_milli = int(nxt())
    switch_cost_i100 = int(nxt())
    cap = int(nxt())
    c = [int(nxt()) for _ in range(K)]
    s = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            v = int(nxt())
            s[i][j] = v
            s[j][i] = v
    thr = [int(nxt()) for _ in range(K)]
    pen = [int(nxt()) for _ in range(K)]
    shelf = [int(nxt()) for _ in range(K)]
    M0_raw = [int(nxt()) for _ in range(K)]
    M0 = [v / 1000.0 for v in M0_raw]
    arr = [[int(nxt()) for _ in range(K)] for _ in range(T)]
    n_spikes = int(nxt())
    spike = [[None] * K for _ in range(T)]
    for _ in range(n_spikes):
        d = int(nxt()); typ = int(nxt()); amt = int(nxt()); sh = int(nxt())
        if 0 <= d < T and 0 <= typ < K:
            spike[d][typ] = (amt, sh)
    switch_cost = switch_cost_i100 / 100.0
    return dict(T=T, K=K, alpha_milli=alpha_milli, switch_cost=switch_cost, cap=cap,
                c=c, s=s, thr=thr, pen=pen, shelf=shelf, M0=M0, arr=arr, spike=spike)


def parse_solution(path, T, K):
    toks = read_tokens(path)
    expected = T * K
    if len(toks) != expected:
        fail(f"expected exactly {expected} numbers (T*K), got {len(toks)}")
    feed = []
    pos = 0
    for t in range(T):
        row = []
        for k in range(K):
            v = parse_float_strict(toks[pos], f"feed[{t}][{k}]")
            pos += 1
            row.append(v)
        feed.append(row)
    return feed


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    inf, outf = sys.argv[1], sys.argv[2]
    inst = parse_instance(inf)
    T, K = inst["T"], inst["K"]
    feed = parse_solution(outf, T, K)

    F, ok, reason = sc.simulate(K, inst["c"], inst["s"], inst["thr"], inst["pen"],
                                 inst["shelf"], inst["alpha_milli"], inst["switch_cost"],
                                 inst["cap"], inst["M0"], inst["arr"], feed, inst["spike"])
    if not ok:
        fail(reason)

    base_feed = sc.baseline_feed(K, T, inst["shelf"], inst["arr"], inst["spike"], inst["cap"])
    B, okb, reasonb = sc.simulate(K, inst["c"], inst["s"], inst["thr"], inst["pen"],
                                   inst["shelf"], inst["alpha_milli"], inst["switch_cost"],
                                   inst["cap"], inst["M0"], inst["arr"], base_feed, inst["spike"])
    if not okb:
        fail("internal: baseline construction infeasible: " + reasonb)

    sc_score = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc_score / 1000.0
    print(f"F={F:.6f} B={B:.6f}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
