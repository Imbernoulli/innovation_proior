#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (<ans> is an unused placeholder)

Replays a rack-management script against the day's request sequence and
scores it.  Prints exactly one line ending in "Ratio: <float in [0,1]>".
"""
import sys


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_from_text(text):
    toks = text.split()
    out = []
    for tk in toks:
        try:
            out.append(int(tk))
        except ValueError:
            return None  # non-integer token (covers nan/inf/garbage)
    return out


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_toks = f.read().split()
    ptr = 0
    n = int(in_toks[ptr]); ptr += 1
    m = int(in_toks[ptr]); ptr += 1
    K = int(in_toks[ptr]); ptr += 1
    P = int(in_toks[ptr]); ptr += 1
    for _ in range(P):
        ptr += 2  # pair list is informational for the solver; not needed to replay
    seq = [int(x) for x in in_toks[ptr:ptr + m]]
    ptr += m
    if len(seq) != m:
        fail("corrupt input instance")
    for x in seq:
        if not (1 <= x <= n):
            fail("corrupt input instance")

    # --- internal baseline B: identity order, zero swaps -------------------
    id_pos = {item: item for item in range(1, n + 1)}  # identity permutation
    B = sum(id_pos[x] for x in seq)
    if B <= 0:
        fail("degenerate instance")

    # --- read participant output defensively --------------------------------
    try:
        with open(out_path, "r") as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    toks_text = raw.split()
    # bounded read of the initial permutation (exactly n tokens)
    if len(toks_text) < n:
        fail("truncated output: missing initial permutation")
    perm_toks = read_ints_from_text(" ".join(toks_text[:n]))
    if perm_toks is None or len(perm_toks) != n:
        fail("non-integer token in initial permutation")
    perm = perm_toks
    if sorted(perm) != list(range(1, n + 1)):
        fail("initial line is not a permutation of 1..n")

    idx = n  # token index cursor into toks_text
    if idx >= len(toks_text):
        fail("missing swap-budget count T")
    t_tok = read_ints_from_text(toks_text[idx])
    if t_tok is None:
        fail("non-integer T")
    T = t_tok[0]
    idx += 1
    if T < 0 or T > K:
        fail("swap budget exceeded (or negative T)")

    need = 2 * T
    if len(toks_text) - idx < need:
        fail("truncated output: missing swap events")
    ev_toks = read_ints_from_text(" ".join(toks_text[idx:idx + need])) if need > 0 else []
    if need > 0 and (ev_toks is None or len(ev_toks) != need):
        fail("non-integer token in swap events")
    events = []
    last_i = 0
    for k in range(T):
        ci = ev_toks[2 * k]
        cj = ev_toks[2 * k + 1]
        if not (1 <= ci <= m):
            fail("swap event customer index out of range")
        if not (1 <= cj <= n - 1):
            fail("swap event slot index out of range")
        if ci < last_i:
            fail("swap events not given in non-decreasing customer-index order")
        last_i = ci
        events.append((ci, cj))

    # --- replay ---------------------------------------------------------
    order = [0] + perm[:]      # order[slot] = item, 1-indexed slots
    pos = [0] * (n + 1)        # pos[item] = slot
    for slot in range(1, n + 1):
        pos[order[slot]] = slot

    F_positions = 0
    ei = 0
    for i in range(1, m + 1):
        while ei < T and events[ei][0] == i:
            j = events[ei][1]
            u, v = order[j], order[j + 1]
            order[j], order[j + 1] = v, u
            pos[u], pos[v] = j + 1, j
            ei += 1
        F_positions += pos[seq[i - 1]]

    if ei != T:
        fail("swap event scheduled after the last customer")

    F = F_positions + T
    if F <= 0:
        fail("degenerate cost")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%d B=%d T=%d" % (F, B, T))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
