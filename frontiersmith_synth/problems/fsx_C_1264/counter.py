#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for compiler-inline-budget.

Instance: m call sites. Call site i has:
  freq_i        dynamic execution count over the run
  base_cost_i   per-invocation body instruction cost (executed whether inlined or not)
  inline_size_i static code bytes ADDED to the caller's image if i is inlined
  parent_i      0 (no requirement) or the 1-based index of an EARLIER call site that
                must ALSO be inlined -- together with parent_i's own ancestor chain --
                before i's constant-propagation bonus can fire
  bonus_i       instructions shaved off base_cost_i, per invocation, once unlocked

Participant artifact: the SET of call sites to inline -- "k" then k distinct 1-based
indices in [1,m].

Deterministic replay:
  effective_cost_i(x) = base_cost_i - bonus_i if UNLOCKED(i,x) else base_cost_i
    UNLOCKED(i,x): x_i=1 AND parent_i==0, OR x_i=1 AND x_{parent_i}=1 AND UNLOCKED-chain
    (i.e. every ancestor on the path to the root is also inlined -- a broken link
    anywhere upstream means the compiler never sees a compile-time constant at i).
  D(x) = sum_i freq_i * ( effective_cost_i(x) + CALL_OVERHEAD * (1 - x_i) )
  S(x) = S_base + sum_i x_i * inline_size_i
  excess = max(0, S(x) - ICACHE_CAP)
  F(x) = D(x) * (ICACHE_CAP + PENALTY_COEF * excess) // ICACHE_CAP      (integer, exact)
Objective: minimize F(x).

Feasibility: well-formed integers, 0 <= k <= m, indices distinct and in [1,m]. Any
violation -> Ratio: 0.0.

Baseline B: the checker's own reference construction -- sort call sites by
(freq_i * CALL_OVERHEAD) / inline_size_i descending (ties by index ascending), and
greedily accept a call site iff it still fits under ICACHE_CAP -- a size-aware but
CHAIN-BLIND frequency recipe that never looks at parent/bonus at all. Minimization
ratio: sc = min(1000, 100*B/F); print(sc/1000).
"""
import sys

MAX_TOKEN_LEN = 20
MAX_K = 100_000


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


def parse_int(tok):
    if len(tok) > MAX_TOKEN_LEN:
        raise ValueError("token too long")
    return int(tok)


def unlocked_mask(chosen, parent, m):
    """chosen: set of 1-based inlined indices. Returns dict i -> bool 'bonus fires'."""
    # memoized ancestor-chain check
    ok_cache = {}

    def chain_ok(i):
        if i in ok_cache:
            return ok_cache[i]
        if i not in chosen:
            ok_cache[i] = False
            return False
        p = parent[i]
        if p == 0:
            ok_cache[i] = True
            return True
        result = chain_ok(p)
        ok_cache[i] = result
        return result

    unlocked = {}
    for i in range(1, m + 1):
        if i in chosen and parent[i] != 0:
            unlocked[i] = chain_ok(parent[i])
        else:
            unlocked[i] = False
    return unlocked


def simulate(chosen, freq, base_cost, inline_size, parent, bonus, m,
             S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF):
    unlocked = unlocked_mask(chosen, parent, m)
    D = 0
    for i in range(1, m + 1):
        eff = base_cost[i]
        if unlocked[i]:
            eff -= bonus[i]
            if eff < 1:
                eff = 1
        overhead = 0 if i in chosen else CALL_OVERHEAD
        D += freq[i] * (eff + overhead)
    S = S_base + sum(inline_size[i] for i in chosen)
    excess = S - ICACHE_CAP
    if excess < 0:
        excess = 0
    F = (D * (ICACHE_CAP + PENALTY_COEF * excess)) // ICACHE_CAP
    return F, D, S


def reference_baseline(freq, inline_size, m, S_base, ICACHE_CAP, CALL_OVERHEAD):
    order = sorted(range(1, m + 1),
                    key=lambda i: (-(freq[i] * CALL_OVERHEAD) / inline_size[i], i))
    chosen = set()
    S = S_base
    for i in order:
        if S + inline_size[i] <= ICACHE_CAP:
            chosen.add(i)
            S += inline_size[i]
    return chosen


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = iter(itoks)

    def inext():
        return next(ip)

    try:
        m = parse_int(inext())
        if m <= 0 or m > 5000:
            fail("bad m (should not happen)")
        S_base = parse_int(inext())
        ICACHE_CAP = parse_int(inext())
        CALL_OVERHEAD = parse_int(inext())
        PENALTY_COEF = parse_int(inext())
        if S_base <= 0 or ICACHE_CAP <= 0 or CALL_OVERHEAD <= 0 or PENALTY_COEF <= 0:
            fail("bad instance header (should not happen)")
        freq = [0] * (m + 1)
        base_cost = [0] * (m + 1)
        inline_size = [0] * (m + 1)
        parent = [0] * (m + 1)
        bonus = [0] * (m + 1)
        for i in range(1, m + 1):
            freq[i] = parse_int(inext())
            base_cost[i] = parse_int(inext())
            inline_size[i] = parse_int(inext())
            parent[i] = parse_int(inext())
            bonus[i] = parse_int(inext())
            if freq[i] <= 0 or base_cost[i] <= 0 or inline_size[i] <= 0:
                fail("bad call site (should not happen)")
            if not (0 <= parent[i] < i):
                fail("bad parent pointer (should not happen)")
            if parent[i] == 0 and bonus[i] != 0:
                fail("root call site has nonzero bonus (should not happen)")
            if bonus[i] < 0 or bonus[i] >= base_cost[i]:
                fail("bad bonus magnitude (should not happen)")
    except (StopIteration, ValueError):
        fail("malformed instance (should not happen)")

    # ---- parse participant output (untrusted) ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except OSError:
        fail("cannot read output")

    if not otoks:
        fail("empty output")

    op = iter(otoks)
    try:
        k = parse_int(next(op))
    except (StopIteration, ValueError):
        fail("bad inline-set size (non-finite/garbage)")
    if k < 0 or k > MAX_K or k > m:
        fail("inline-set size out of range")

    chosen_list = []
    try:
        for _ in range(k):
            idx = parse_int(next(op))
            chosen_list.append(idx)
    except (StopIteration, ValueError):
        fail("malformed call-site index (non-finite/garbage/truncated)")

    seen = set()
    for idx in chosen_list:
        if not (1 <= idx <= m):
            fail("call-site index out of range [1,m]")
        if idx in seen:
            fail("duplicate call-site index")
        seen.add(idx)

    # extra trailing garbage tokens are ignored (schema is fixed-length, self-delimited)

    # ---- score ----
    F, D, S = simulate(seen, freq, base_cost, inline_size, parent, bonus, m,
                        S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF)
    if F <= 0:
        fail("degenerate replay (should not happen)")

    base_chosen = reference_baseline(freq, inline_size, m, S_base, ICACHE_CAP, CALL_OVERHEAD)
    B, _, _ = simulate(base_chosen, freq, base_cost, inline_size, parent, bonus, m,
                        S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF)
    if B <= 0:
        fail("degenerate baseline (should not happen)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
