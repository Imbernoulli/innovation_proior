#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>

Deterministic checker for the reaction-network firing-sequence problem.

<in>  : m N_R N_Z
        m lines: px py dx dv cx cy cw   (one per group g=0..m-1)

<out> : line 1: K (number of firings)
        then K whitespace-separated tokens, each "<g><L>" where g is a
        group index (0..m-1) and L in {X,Y,D,C} selects the reaction:
          gX : px_g R            -> 1 X_g
          gY : py_g R            -> 1 Y_g
          gD : dx_g X_g          -> dv_g T
          gC : cx_g X_g + cy_g Y_g + 1 Z -> cw_g T

Feasibility: every firing must be affordable from CURRENT stock at the
moment it fires (irreversible consumption-depletion); any shortfall ->
Ratio: 0.0.

Score: F = final T count achieved by the submitted sequence.
Baseline B = a trivial fixed feasible construction the checker builds
itself: spend all of N_R on group 0's direct route ONLY (ignore every
other group, ignore combos/catalyst entirely).
  sc = min(1000.0, 100.0*F/max(1e-9,B));  print Ratio: sc/1000
"""
import sys
import re

TOKEN_RE = re.compile(r"^(\d+)([XYDC])$")
MAX_TOKENS = 2_000_000


def fail(reason):
    print("Infeasible/invalid: %s Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tokens = f.read().split()
    try:
        idx = 0
        m = int(in_tokens[idx]); idx += 1
        N_R = int(in_tokens[idx]); idx += 1
        N_Z = int(in_tokens[idx]); idx += 1
        px = [0] * m; py = [0] * m; dx = [0] * m; dv = [0] * m
        cx = [0] * m; cy = [0] * m; cw = [0] * m
        for g in range(m):
            px[g] = int(in_tokens[idx]); idx += 1
            py[g] = int(in_tokens[idx]); idx += 1
            dx[g] = int(in_tokens[idx]); idx += 1
            dv[g] = int(in_tokens[idx]); idx += 1
            cx[g] = int(in_tokens[idx]); idx += 1
            cy[g] = int(in_tokens[idx]); idx += 1
            cw[g] = int(in_tokens[idx]); idx += 1
    except Exception:
        fail("could not parse instance file")
        return

    # ---- read participant output ----
    try:
        with open(out_path) as f:
            content = f.read()
    except Exception:
        fail("missing output file")
        return

    out_tokens = content.split()
    if len(out_tokens) == 0:
        fail("empty output")
        return

    try:
        K = int(out_tokens[0])
    except Exception:
        fail("first token is not an integer K")
        return

    if K < 0 or K > MAX_TOKENS:
        fail("K out of allowed range")
        return

    body = out_tokens[1:]
    if len(body) != K:
        fail("declared K does not match number of firing tokens present")
        return

    # ---- validate + parse every firing token strictly ----
    fires = []
    for tok in body:
        mobj = TOKEN_RE.match(tok)
        if not mobj:
            fail("malformed firing token %r" % tok)
            return
        g = int(mobj.group(1))
        letter = mobj.group(2)
        if g < 0 or g >= m:
            fail("group index out of range in token %r" % tok)
            return
        fires.append((g, letter))

    # ---- simulate irreversible consumption/production ----
    R = N_R
    Z = N_Z
    X = [0] * m
    Y = [0] * m
    T = 0

    for step, (g, letter) in enumerate(fires):
        if letter == "X":
            if R < px[g]:
                fail("step %d: insufficient R to fire %dX" % (step, g))
                return
            R -= px[g]
            X[g] += 1
        elif letter == "Y":
            if R < py[g]:
                fail("step %d: insufficient R to fire %dY" % (step, g))
                return
            R -= py[g]
            Y[g] += 1
        elif letter == "D":
            if X[g] < dx[g]:
                fail("step %d: insufficient X_%d to fire %dD" % (step, g, g))
                return
            X[g] -= dx[g]
            T += dv[g]
        elif letter == "C":
            if X[g] < cx[g] or Y[g] < cy[g] or Z < 1:
                fail("step %d: insufficient precursors/catalyst to fire %dC" % (step, g))
                return
            X[g] -= cx[g]
            Y[g] -= cy[g]
            Z -= 1
            T += cw[g]
        else:
            fail("unknown reaction letter")
            return

    F = T
    if not (F == F) or F in (float("inf"), float("-inf")):
        fail("non-finite objective")
        return
    if F < 0:
        fail("negative objective")
        return

    # ---- checker's own trivial baseline: group 0, direct route only ----
    denom0 = dx[0] * px[0]
    f0 = N_R // denom0 if denom0 > 0 else 0
    B = f0 * dv[0]

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
