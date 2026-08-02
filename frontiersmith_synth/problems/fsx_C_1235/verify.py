#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the dual-write
migration-plan problem (family: migration-dual-write).

Feasibility: the participant's plan is replayed against the fixed input
timeline. ANY read-check tick at/after the chosen cutover that would observe
a value in the NEW store different from the true (OLD-store) value is a lost
update -> the whole test case scores 0.0.

Objective (on a feasible plan):
    earliness    = (T - C) / T                         in [0, 1]
    completeness = (# keys whose NEW value == TRUE value at the final tick) / K
    F = ALPHA * earliness + BETA * completeness

Baseline B (checker's own safe reference): "never cut over, protect every
backfill write with a version check" -- always feasible (no reads are ever
gated) and, given the input's version/watermark invariants, always fully
correct at the end, so completeness = 1 and earliness = 0 -> B = BETA.
(We still compute it by simulation rather than hard-coding, matching the
general-purpose engine below.)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    Ratio = sc / 1000.0
"""
import sys, math

ALPHA = 9.0
BETA = 1.0


def die0(reason):
    print(f"INFEASIBLE: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split('\n')
    toks = [ln for ln in toks if ln.strip() != '']
    head = toks[0].split()
    K, T, M = int(head[0]), int(head[1]), int(head[2])
    baseline = list(map(int, toks[1].split()))
    ops = []
    for ln in toks[2:2 + T]:
        parts = ln.split()
        if parts[0] == 'R':
            ops.append(('R', int(parts[1])))
        else:
            ops.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
    return K, T, M, baseline, ops


def _finite_num(tok):
    """Parse tok as a finite float; return None on failure/non-finite."""
    try:
        v = float(tok)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_output(path, K, T, M):
    with open(path) as f:
        raw = f.read()
    toks = raw.split()
    if len(toks) != 1 + M:
        return None, f"expected {1 + M} tokens (1 cutover + {M} flags), got {len(toks)}"
    c_num = _finite_num(toks[0])
    if c_num is None:
        return None, "cutover tick is not a finite number"
    if abs(c_num - round(c_num)) > 1e-9:
        return None, "cutover tick is not an integer"
    C = int(round(c_num))
    if not (0 <= C <= T):
        return None, f"cutover tick {C} out of range [0,{T}]"
    flags = []
    for tok in toks[1:]:
        v = _finite_num(tok)
        if v is None:
            return None, "a backfill flag is not a finite number"
        if abs(v - round(v)) > 1e-9 or round(v) not in (0, 1):
            return None, "a backfill flag is not exactly 0 or 1"
        flags.append(int(round(v)))
    return (C, flags), None


def simulate(K, T, ops, baseline, C, flags):
    """Returns (feasible: bool, reason, F: float)."""
    NEW = {k: None for k in range(1, K + 1)}      # None or (version, value)
    TRUE = {k: (0, baseline[k - 1]) for k in range(1, K + 1)}
    bi = 0
    for i, op in enumerate(ops):
        if op[0] == 'L':
            _, k, v, x = op
            TRUE[k] = (v, x)
            NEW[k] = (v, x)                        # dual-write always lands
        elif op[0] == 'B':
            _, k, v, x = op
            f = flags[bi]; bi += 1
            cur = NEW[k]
            if f == 1:
                if cur is None or v > cur[0]:
                    NEW[k] = (v, x)
                # else: conditional write correctly no-ops on a stale snapshot
            else:
                NEW[k] = (v, x)                    # unconditional: always overwrites
        else:  # 'R'
            _, k = op
            if i >= C:
                cur = NEW[k]
                if cur is None or cur != TRUE[k]:
                    return False, f"read-check on key {k} at tick {i} (cutover={C}) observed a stale/missing value in the NEW store", 0.0
    completeness = sum(1 for k in range(1, K + 1) if NEW[k] == TRUE[k]) / K
    earliness = (T - C) / T
    F = ALPHA * earliness + BETA * completeness
    return True, "", F


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0"); return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    K, T, M, baseline, ops = read_input(in_path)

    parsed, err = parse_output(out_path, K, T, M)
    if parsed is None:
        die0(err)
    C, flags = parsed

    feasible, reason, F = simulate(K, T, ops, baseline, C, flags)
    if not feasible:
        die0(reason)

    # checker's own reference: never cut over, always CAS-protect every backfill
    _, _, B = simulate(K, T, ops, baseline, T, [1] * M)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print(f"F={F:.6f} B={B:.6f}")
    print(f"Ratio: {sc / 1000.0:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
