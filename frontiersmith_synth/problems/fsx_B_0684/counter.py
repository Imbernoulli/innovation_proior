#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for shoulder-shadow-turning.

Verifies the submitted turning tape (sequence of single-band passes) against:
  - machining precedence DAG (a band's neighbour with the LARGER target must be
    fully finished before this band's own FINAL pass),
  - the overhang-chatter limit  x_band * depth <= K * S(state), where
    S(state) = min(current[1..band]) is the "weakest link" support stiffness,
  - per-tool [minDepth, maxDepth] legality,
  - exact final-profile equivalence (no undercut, no leftover material),
then counts a total op-cost = (#passes) + C * (#tool-change events) and reports
Ratio = min(1, B / F) against an internally-built, always-feasible baseline B.
"""
import sys

MAX_P = 200000
MAX_TOKLEN = 18


def bail(msg):
    print(f"# {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    try:
        with open(path, "r") as f:
            return f.read().split()
    except Exception:
        return None


def parse_int(tok):
    if tok is None or len(tok) > MAX_TOKLEN:
        return None
    try:
        return int(tok)
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        bail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_tokens(in_path)
    if not itoks or len(itoks) < 5:
        bail("bad input file")
    it = iter(itoks)
    n = parse_int(next(it)); m = parse_int(next(it)); K = parse_int(next(it))
    C = parse_int(next(it)); R = parse_int(next(it))
    if None in (n, m, K, C, R) or n <= 0 or m <= 0 or K <= 0 or C < 0 or R <= 1:
        bail("bad header")
    target = []
    for _ in range(n):
        v = parse_int(next(it, None))
        if v is None:
            bail("bad target token")
        target.append(v)
    for v in target:
        if not (1 <= v <= R - 1):
            bail("target out of range")
    tools = []
    for _ in range(m):
        mx = parse_int(next(it, None)); mn = parse_int(next(it, None))
        if mx is None or mn is None or not (1 <= mn <= mx):
            bail("bad tool spec")
        tools.append((mx, mn))

    otoks = read_tokens(out_path)
    if otoks is None or len(otoks) == 0:
        bail("empty/missing output")
    oit = iter(otoks)
    P = parse_int(next(oit, None))
    if P is None or P < 0 or P > MAX_P:
        bail("bad pass count")
    if len(otoks) != 1 + 3 * P:
        bail("token count mismatch")

    passes = []
    for _ in range(P):
        tid = parse_int(next(oit, None))
        band = parse_int(next(oit, None))
        depth = parse_int(next(oit, None))
        if tid is None or band is None or depth is None:
            bail("bad pass token")
        passes.append((tid, band, depth))

    current = list(target)
    for i in range(n):
        current[i] = R  # start full stock

    for (tid, band, depth) in passes:
        if not (1 <= tid <= m):
            bail(f"bad tool id {tid}")
        if not (1 <= band <= n):
            bail(f"bad band {band}")
        mx, mn = tools[tid - 1]
        if not (mn <= depth <= mx):
            bail("depth outside tool's [minDepth,maxDepth]")
        i0 = band - 1
        headroom = current[i0] - target[i0]
        if depth > headroom or headroom <= 0:
            bail(f"undercut/no-material at band {band}")

        is_final = (depth == headroom)
        if is_final:
            # neighbours with strictly larger target must already be finished
            for j0 in (i0 - 1, i0 + 1):
                if 0 <= j0 < n and target[j0] > target[i0]:
                    if current[j0] != target[j0]:
                        bail(f"precedence violated finishing band {band}")

        # chatter: x_band * depth <= K * S ,  S = min(current[1..band]) BEFORE this pass
        S = min(current[0:band])
        x = band
        if x * depth > K * S:
            bail(f"chatter limit violated at band {band}")

        current[i0] -= depth

    for i in range(n):
        if current[i] != target[i]:
            bail(f"band {i+1} not finished (final={current[i]}, target={target[i]})")

    tool_changes = 0
    prev = None
    for (tid, _, _) in passes:
        if tid != prev:
            tool_changes += 1
        prev = tid

    F = P + C * tool_changes

    # ---- internal baseline: always-feasible, single-tool, unit-depth construction ----
    # order = descending target (ties by ascending index) respects EVERY pairwise
    # precedence edge; depth=1 with any minDepth==1 tool is always chatter-legal
    # because target_i>=1 => S>=1 always, and K>=n by construction => K*S>=n>=x.
    fin_tool = None
    for idx, (mx, mn) in enumerate(tools):
        if mn == 1:
            fin_tool = idx + 1
            break
    if fin_tool is None:
        fin_tool = 1  # defensive; generator always provides a minDepth==1 tool
    mx_fin, mn_fin = tools[fin_tool - 1]

    base_order = sorted(range(n), key=lambda i: (-target[i], i))
    cur = [R] * n
    base_passes = 0
    for i in base_order:
        band = i + 1
        while cur[i] > target[i]:
            headroom = cur[i] - target[i]
            S = min(cur[0:band])
            cap_chatter = (K * S) // band
            depth = min(mx_fin, headroom, cap_chatter)
            if depth < 1:
                depth = 1  # guaranteed legal: target_i>=1 => S>=1, K>=n => K*S>=n>=band
            cur[i] -= depth
            base_passes += 1
    B = base_passes + C * 1  # one tool throughout, one tool-change event (loading it)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"# P={P} tool_changes={tool_changes} F={F} B={B}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
