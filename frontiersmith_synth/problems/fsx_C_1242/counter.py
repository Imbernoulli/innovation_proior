#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for the systolic-array
dataflow-mapping problem (Format D, eval_form=flops).

Pipeline:
  1. Read the instance: array P x Q, L matmul layers (M_i,K_i,N_i), a per-tile
     reload cost RELOAD and a per-switch reconfiguration cost SWITCH.
  2. Parse the participant's artifact under a strict, bounded schema: a header
     line repeating L, then exactly L lines each one of the 6 permutations of
     the letters "M","K","N" (which dimension maps to the array's row axis,
     which to the column axis, which streams). ANY structural violation
     (wrong header, wrong line count, unknown/duplicated letters, extra
     tokens, non-finite header) -> Ratio: 0.0. This is this family's
     "exact-equivalence" gate: the artifact must be a well-formed dataflow
     schedule before its cost counts at all.
  3. Compute F = total scalar-cell-cycles (the FLOPs/op-count surrogate):
     for each layer, tiles-to-cover-the-stationary-plane * P*Q (every array
     cell is charged for every cycle it's powered, valid data or not) *
     (RELOAD + stream_dim + pipeline-fill/drain); plus SWITCH*P*Q whenever
     consecutive layers use different codes.
  4. Internal baseline B = the exact same formula, forcing the single fixed
     code "KNM" (classic weight-stationary: pin K,N; stream M) on every
     layer regardless of shape -- a legitimate, shape-blind construction.
     Minimization: Ratio = min(1.0, 0.1 * B / max(1e-9, F)).
Everything is exact integer arithmetic; nothing is timed and there is no
randomness.
"""
import sys

CODES = {"MKN", "MNK", "KMN", "KNM", "NMK", "NKM"}
MAX_LINE = 64
MAX_HEADER = 10 ** 15  # bound before int() to avoid pathological bignum parses


def read_instance(path):
    toks = open(path).read().split()
    i = 0

    def nxt():
        nonlocal i
        v = toks[i]
        i += 1
        return v

    P = int(nxt()); Q = int(nxt()); L = int(nxt())
    RELOAD = int(nxt()); SWITCH = int(nxt())
    layers = []
    for _ in range(L):
        m = int(nxt()); k = int(nxt()); n = int(nxt())
        layers.append({'M': m, 'K': k, 'N': n})
    return {'P': P, 'Q': Q, 'L': L, 'RELOAD': RELOAD, 'SWITCH': SWITCH, 'layers': layers}


def layer_cost(P, Q, RELOAD, dims, code):
    d1, d2, s = code[0], code[1], code[2]
    D1, D2, S = dims[d1], dims[d2], dims[s]
    tp = -(-D1 // P)   # ceil
    tq = -(-D2 // Q)
    pipe = P + Q - 1
    per_tile = RELOAD + S + pipe
    return tp * tq * P * Q * per_tile


def total_cost(inst, codes):
    P, Q, RELOAD, SWITCH = inst['P'], inst['Q'], inst['RELOAD'], inst['SWITCH']
    total = 0
    prev = None
    for dims, code in zip(inst['layers'], codes):
        total += layer_cost(P, Q, RELOAD, dims, code)
        if prev is not None and code != prev:
            total += SWITCH * P * Q
        prev = code
    return total


def parse_artifact(text, L):
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        return False, "empty output", None
    if len(lines) != L + 1:
        return False, f"expected {L + 1} non-blank lines, got {len(lines)}", None
    header_toks = lines[0].split()
    if len(header_toks) != 1:
        return False, "header line must be a single integer L", None
    htxt = header_toks[0]
    # strict integer token: optional leading '-', digits only (rejects nan/inf/floats)
    body = htxt[1:] if htxt.startswith('-') else htxt
    if body == "" or not body.isdigit() or len(body) > 15:
        return False, f"bad header token: '{htxt}'", None
    header = int(htxt)
    if header != L:
        return False, f"header {header} != L {L}", None
    codes = []
    for ln in lines[1:]:
        if len(ln) > MAX_LINE:
            return False, "code line too long", None
        toks = ln.split()
        if len(toks) != 1:
            return False, f"code line must be a single token: '{ln}'", None
        c = toks[0]
        if c not in CODES:
            return False, f"'{c}' is not a valid code (permutation of M,K,N)", None
        codes.append(c)
    return True, "", codes


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = read_instance(in_path)
    L = inst['L']

    try:
        text = open(out_path, 'r', errors='replace').read()
    except Exception:
        print("INVALID: cannot read output")
        print("Ratio: 0.0")
        return

    if len(text) > 20000:
        print("INVALID: output too large")
        print("Ratio: 0.0")
        return

    ok, reason, codes = parse_artifact(text, L)
    if not ok:
        print(f"INVALID: {reason}")
        print("Ratio: 0.0")
        return

    F = total_cost(inst, codes)
    B = total_cost(inst, ["KNM"] * L)

    if F != F or F in (float('inf'), float('-inf')) or F <= 0:
        print("INVALID: non-finite or non-positive cost")
        print("Ratio: 0.0")
        return

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
