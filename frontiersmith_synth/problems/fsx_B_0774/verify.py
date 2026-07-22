#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  (ans unused, empty placeholder)

Checker for 'word-autocorrelation-realize'.
Output artifact: exactly n whitespace-separated integer letter-indices in
[0,K). Feasibility: token count == n, every token a valid integer in [0,K).
Score: matched target-border weight - lambda*spurious-border-count
       + alpha*(K - distinct-letters-used) + 1, normalized against an
       internal weak (golden-ratio rotation) baseline.
"""
import sys, math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    lam = int(toks[idx]); idx += 1
    alpha = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    targets = []
    for _ in range(m):
        b = int(toks[idx]); idx += 1
        w = int(toks[idx]); idx += 1
        targets.append((b, w))
    return n, K, lam, alpha, targets


def actual_borders(W):
    # W: tuple/list of ints
    n = len(W)
    res = set()
    for b in range(1, n):
        if W[:b] == W[n - b:]:
            res.add(b)
    return res


def score_word(W, n, K, lam, alpha, targets):
    target_set = set(b for b, w in targets)
    ab = actual_borders(W)
    matched = sum(w for b, w in targets if b in ab)
    spurious = len(ab - target_set)
    distinct = len(set(W))
    F = matched - lam * spurious + alpha * max(0, K - distinct) + 2
    return F


def baseline_word(n, K):
    # Target-blind reference: a low-discrepancy (golden-ratio) rotation sequence.
    # Unlike a short-period cyclic fill, this is aperiodic-in-practice at these
    # scales, so it does not accidentally manufacture a pile of spurious
    # borders -- it stays a genuinely WEAK (but not self-sabotaging) baseline.
    phi = 1.6180339887498949
    return tuple(int(((i + 1) * phi) % 1.0 * K) for i in range(n))


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, K, lam, alpha, targets = read_instance(in_path)

    try:
        raw = open(out_path).read()
    except Exception:
        fail("cannot read output")

    otoks = raw.split()
    if len(otoks) != n:
        fail("expected exactly n=%d integer tokens (the word), got %d" % (n, len(otoks)))

    W = []
    for t in otoks:
        try:
            v = int(t)
        except ValueError:
            fail("non-integer token %r in output" % t)
        if not (0 <= v < K):
            fail("letter index %d outside allowed alphabet [0,%d)" % (v, K))
        W.append(v)
    W = tuple(W)

    F = score_word(W, n, K, lam, alpha, targets)

    bw = baseline_word(n, K)
    F_base_raw = score_word(bw, n, K, lam, alpha, targets)
    B = max(1.0, float(F_base_raw))

    SCALE = 18.0  # headroom factor: F == SCALE*B is required to saturate the ratio at 1.0
    sc = min(1000.0, (1000.0 / SCALE) * F / max(1e-9, B))
    sc = max(0.0, sc)
    ratio = sc / 1000.0
    print("matched/spurious/distinct info: F=%.3f baseline=%.3f" % (F, B))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
