#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for the irrigation-split
weather-book problem.

Feasibility (strict; any violation -> Ratio: 0.0):
  - output starts with "F T" matching the input dims exactly
  - exactly F*T further tokens follow (no more, no fewer)
  - every token parses as a finite non-negative integer <= 10_000_000
  - every week's cross-field irrigation sum <= WeeklyCap
  - the grand total irrigation <= TotalBudget

Objective (exact integer arithmetic, no floats until the final ratio):
  For field f under scenario k, simulate the weekly soil-moisture recurrence
      m <- clip(m + X[f][t] + rain[k][t] - cons[f], 0, cap[f])
  and accrue growth = rate[f]*m in weeks where m >= thresh[f] (0 in a
  stressed week -- the crop stalls that week but can recover later).
  scenario_yield(k) = sum over fields of that field's season total.
  OBJECTIVE = PRODUCT over scenarios of scenario_yield(k)  (an unhedged
  weak scenario multiplies the whole product down -- product-of-scenario-
  yields).  The checker also builds its own feasible uniform-split baseline
  B and reports the same objective on it; score = min(1, 0.1*OBJECTIVE/B)
  scaled per the maximization convention below.
"""
import sys
from fractions import Fraction

MAX_TOKEN = 10_000_000


def die(msg):
    print(f"# {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as fh:
        toks = fh.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = toks[idx]
        idx += 1
        return v

    T = int(nxt()); F = int(nxt()); K = int(nxt())
    total_budget = int(nxt()); weekly_cap = int(nxt())
    fields = []
    for _ in range(F):
        m0 = int(nxt()); cap = int(nxt()); cons = int(nxt())
        thresh = int(nxt()); rate = int(nxt())
        fields.append((m0, cap, cons, thresh, rate))
    rain = []
    for _ in range(K):
        rain.append([int(nxt()) for _ in range(T)])
    return T, F, K, total_budget, weekly_cap, fields, rain


def simulate_objective(X, T, F, K, fields, rain):
    """Exact-integer PRODUCT over scenarios of the summed field yields.

    Weekly growth has two parts, by design: a FLAT reward `rate*thresh` for
    simply clearing the stress threshold that week (this is the dominant
    term -- avoiding stress is what matters), plus a small capped BONUS for
    extra moisture above threshold (a modest, secondary optimization
    opportunity for any leftover budget once every scenario is safe). A
    stressed week (m < thresh) earns nothing at all -- the crop stalls.
    """
    product = 1
    for k in range(K):
        rk = rain[k]
        scenario_sum = 0
        for f in range(F):
            m0, cap, cons, thresh, rate = fields[f]
            m = m0
            Xf = X[f]
            total = 0
            half_thresh = thresh // 2
            for t in range(T):
                m = m + Xf[t] + rk[t] - cons
                if m < 0:
                    m = 0
                elif m > cap:
                    m = cap
                if m >= thresh:
                    margin = min(m - thresh, half_thresh)
                    total += rate * thresh + (rate * margin) // 4
            scenario_sum += total
        product *= scenario_sum
    return product


def main():
    if len(sys.argv) < 3:
        die("bad checker invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    T, F, K, total_budget, weekly_cap, fields, rain = read_input(in_path)

    try:
        with open(out_path) as fh:
            out_toks = fh.read().split()
    except Exception:
        die("cannot read output")

    if len(out_toks) < 2:
        die("output too short")

    def parse_int_strict(tok):
        # reject anything that is not a plain finite integer literal
        # (blocks nan/inf/float garbage from masquerading as ints)
        s = tok
        neg = s.startswith("-")
        body = s[1:] if neg else s
        if not body.isdigit():
            raise ValueError("not an integer token: %r" % tok)
        return int(s)

    try:
        F_out = parse_int_strict(out_toks[0])
        T_out = parse_int_strict(out_toks[1])
    except Exception as e:
        die(f"bad header: {e}")

    if F_out != F or T_out != T:
        die(f"dims mismatch: got F={F_out} T={T_out}, expected F={F} T={T}")

    rest = out_toks[2:]
    if len(rest) != F * T:
        die(f"expected exactly {F * T} irrigation tokens, got {len(rest)}")

    try:
        vals = [parse_int_strict(t) for t in rest]
    except Exception as e:
        die(f"bad irrigation token: {e}")

    for v in vals:
        if v < 0 or v > MAX_TOKEN:
            die(f"irrigation value out of range: {v}")

    X = [vals[f * T:(f + 1) * T] for f in range(F)]

    total_used = sum(vals)
    if total_used > total_budget:
        die(f"total irrigation {total_used} exceeds budget {total_budget}")

    for t in range(T):
        wk = sum(X[f][t] for f in range(F))
        if wk > weekly_cap:
            die(f"week {t} irrigation {wk} exceeds weekly cap {weekly_cap}")

    Y_sub = simulate_objective(X, T, F, K, fields, rain)

    # checker's own feasible constructive baseline: split evenly across
    # every (field, week) slot, respecting both caps by construction.
    per_slot = min(total_budget // (F * T), weekly_cap // F) if F * T > 0 and F > 0 else 0
    per_slot = max(per_slot, 0)
    X_base = [[per_slot] * T for _ in range(F)]
    Y_base = simulate_objective(X_base, T, F, K, fields, rain)

    if Y_base <= 0:
        die("degenerate baseline (generator bug)")

    ratio = Fraction(100 * Y_sub, Y_base)
    sc = ratio if ratio < 1000 else Fraction(1000)
    score = float(sc) / 1000.0
    if score < 0.0:
        score = 0.0
    print("Ratio: %.6f" % score)
    sys.exit(0)


if __name__ == "__main__":
    main()
