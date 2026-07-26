#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the
solidification-front microstructure problem. Prints "... Ratio: <float>".
"""
import sys
import math

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import sim


def fail(msg):
    print("INFEASIBLE: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_int_token(tok):
    """Strict integer parse; rejects floats, nan, inf, junk."""
    try:
        if any(c in tok for c in (".", "e", "E", "n", "N", "f", "F", "d", "D")):
            # allow a bare leading '-' for negative ints but reject float-ish
            # tokens (nan/inf/1.5/1e9) outright -- integers never contain these.
            if tok.lstrip("-").isdigit():
                pass
            else:
                return None
        return int(tok)
    except ValueError:
        return None


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        toks = f.read().split()
    p = 0
    N = int(toks[p]); p += 1
    K = int(toks[p]); p += 1
    H0 = int(toks[p]); p += 1
    F = int(toks[p]); p += 1
    CSTEP = int(toks[p]); p += 1
    P = int(toks[p]); p += 1
    seeds = []
    for _ in range(P):
        pos = int(toks[p]); p += 1
        o = int(toks[p]); p += 1
        seeds.append((pos, o))
    T = [int(toks[p + i]) for i in range(N)]
    p += N

    try:
        with open(outf) as f:
            out_txt = f.read()
    except Exception:
        fail("cannot read output")

    raw_toks = out_txt.split()
    if len(raw_toks) != K:
        fail(f"expected exactly {K} tokens (one action per stage), got {len(raw_toks)}")

    actions = []
    for t in raw_toks:
        v = parse_int_token(t)
        if v is None:
            fail(f"non-integer / non-finite token {t!r}")
        if not (-1 <= v <= N - 1):
            fail(f"action {v} out of range [-1,{N-1}]")
        actions.append(v)

    heat, solid, orient = sim.new_state(N, seeds, H0)
    for stage_i, a in enumerate(actions):
        cool_idx = None if a == -1 else a
        ok = sim.step(heat, solid, orient, cool_idx, F, CSTEP)
        if not ok:
            fail(f"stage {stage_i+1}: cooled cell {a} which is out of range or already solid")

    F_raw = sum(1 for i in range(N) if solid[i] and orient[i] == T[i])

    # internal baseline: the checker's own "do nothing" construction (only the
    # seeds are ever solid) -- computed by literally simulating it, not assumed.
    bheat, bsolid, borient = sim.new_state(N, seeds, H0)
    for _ in range(K):
        sim.step(bheat, bsolid, borient, None, F, CSTEP)
    B_raw = sum(1 for i in range(N) if bsolid[i] and borient[i] == T[i])
    B_raw = max(B_raw, 1e-9)

    sc = min(1000.0, 100.0 * F_raw / B_raw)
    ratio = sc / 1000.0
    if not math.isfinite(ratio):
        fail("non-finite ratio computed")
    print(f"matches={F_raw} baseline={B_raw} N={N}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
