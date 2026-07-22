#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1020
   Family: differential-adhesion-sorting (format C, minimize distance-to-target-topology).
"""
import sys, math

MAX_SWEEPS = 60
MAX_TOKENS = 5000


def read_instance(path):
    toks = open(path).read().split()
    L = int(toks[0]); T = int(toks[1])
    Jmax = int(toks[2])
    target_type = int(toks[3])
    idx = 4
    counts = [int(toks[idx]), int(toks[idx + 1]), int(toks[idx + 2])]
    idx += 3
    arr = [int(toks[idx + i]) for i in range(L)]
    return L, T, Jmax, target_type, counts, arr


def parse_matrix(text, T, Jmax):
    """Return (J, reason). J = T x T list of ints, or None on failure."""
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    if len(toks) != T * T:
        return None, f"expected {T*T} tokens, got {len(toks)}"
    vals = []
    for tok in toks:
        try:
            v = int(tok)
        except ValueError:
            return None, "non-integer token (nan/inf/garbage)"
        vals.append(v)
    J = [vals[r * T:(r + 1) * T] for r in range(T)]
    for a in range(T):
        for b in range(T):
            if J[a][b] < -Jmax or J[a][b] > Jmax:
                return None, f"J[{a}][{b}]={J[a][b]} out of range [-{Jmax},{Jmax}]"
    for a in range(T):
        for b in range(T):
            if J[a][b] != J[b][a]:
                return None, f"matrix not symmetric at ({a},{b})"
    return J, "ok"


def run_dynamics(J, arr0, max_sweeps):
    """Deterministic zero-temperature any-pair swap dynamics (fixed sweep order)."""
    arr = list(arr0)
    L = len(arr)

    def local_energy(edge_idxs):
        s = 0
        for e in edge_idxs:
            s += J[arr[e]][arr[(e + 1) % L]]
        return s

    for _sweep in range(max_sweeps):
        changed = False
        for i in range(L):
            for j in range(i + 1, L):
                if arr[i] == arr[j]:
                    continue
                edge_idxs = {(i - 1) % L, i, (j - 1) % L, j}
                before = local_energy(edge_idxs)
                arr[i], arr[j] = arr[j], arr[i]
                after = local_energy(edge_idxs)
                if after - before < 0:
                    changed = True
                else:
                    arr[i], arr[j] = arr[j], arr[i]
        if not changed:
            break
    return arr


def homotypic_count(arr):
    L = len(arr)
    return sum(1 for i in range(L) if arr[i] == arr[(i + 1) % L])


def evaluate(J, arr0, L, T, target_type):
    final = run_dynamics(J, arr0, MAX_SWEEPS)
    H = homotypic_count(final)
    if target_type == 0:
        gap = (L - T) - H
    else:
        gap = H
    gap = max(0, gap)
    F = (gap + 0.5) / (L + 0.5)
    return F, gap, final


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    L, T, Jmax, target_type, counts, arr0 = read_instance(inf)

    text = open(outf).read()
    J, reason = parse_matrix(text, T, Jmax)
    if J is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0

    F, gap, final = evaluate(J, arr0, L, T, target_type)
    if not math.isfinite(F):
        print("non-finite objective")
        print("Ratio: 0.0")
        return 0

    zero_J = [[0] * T for _ in range(T)]
    B, gapB, _ = evaluate(zero_J, arr0, L, T, target_type)
    B = max(B, 1e-6)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"target_type={target_type} gap={gap} F={F:.4f} baseline_gap={gapB} B={B:.4f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
