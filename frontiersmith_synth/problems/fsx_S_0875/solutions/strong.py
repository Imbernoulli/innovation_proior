# TIER: strong
# The insight: don't pool everything into one flexible fit. Run the analysis
# as a DESIGNED EXPERIMENT, using the row TYPE (which is visible from neighbor
# count / spacing pattern, no tag needed):
#
#   1. ISOLATED PAIRS (n=1 neighbor) give a clean 1-D slice of the pairwise
#      kernel with only one term in the sum -- grid-search the softening
#      offset c and fit the kernel scale a by regression-through-origin on
#      the linearized relation  v/y - 1 ~= a / (d^2 + c)  using ONLY pairs.
#   2. CONTROLLED CLUSTERS (near-equal-spacing rings of known size) push S
#      into a genuinely bigger, multi-term regime while the per-neighbor
#      kernel value stays interpretable -- refine c and a against
#      pairs+clusters together, which is what actually reveals whether the
#      aggregate keeps growing linearly or SATURATES.
#   3. A final light polish against ALL rows (pairs+clusters+organic) tightens
#      the scale.
#
# The recovered (a, c) are then plugged into the RECIPROCAL saturating form
# OUT = v / (1 + S) -- not a truncated polynomial -- so it stays bounded and
# extrapolates correctly to the dense held-out swarms.
import sys


def parse(data):
    n_rows = int(data[0])
    idx = 2
    rows = []
    for _ in range(n_rows):
        k = int(data[idx]); v = float(data[idx + 1]); y = float(data[idx + 2])
        dists = [float(x) for x in data[idx + 3: idx + 3 + k]]
        idx += 3 + k
        rows.append((k, v, y, dists))
    return rows


def is_cluster(row):
    k, v, y, dists = row
    if k < 2:
        return False
    m = sum(dists) / len(dists)
    if m <= 1e-9:
        return False
    spread = (max(dists) - min(dists)) / m
    return spread < 0.06


def fit_a_through_origin(rows, c):
    """Linearized regression-through-origin: r_i ~= a * X_i,
    r_i = v_i/y_i - 1, X_i = sum_j 1/(dist_ij^2 + c)."""
    sxx = 0.0
    sxy = 0.0
    for k, v, y, dists in rows:
        if y <= 1e-6 or v <= 1e-9:
            continue
        X = sum(1.0 / (d * d + c) for d in dists)
        r = v / y - 1.0
        sxx += X * X
        sxy += X * r
    if sxx < 1e-12:
        return 0.0
    return max(0.0, sxy / sxx)


def sse_yspace(rows, a, c):
    e = 0.0
    for k, v, y, dists in rows:
        X = sum(1.0 / (d * d + c) for d in dists)
        pred = v / (1.0 + a * X)
        e += (pred - y) ** 2
    return e


def main():
    data = sys.stdin.read().split()
    if not data:
        print("KERNEL 0"); print("OUT v"); return
    rows = parse(data)
    pairs = [r for r in rows if r[0] == 1]
    clusters = [r for r in rows if is_cluster(r)]

    if not pairs:
        pairs = rows

    # Stage 1: isolated pairs only -> coarse (c, a)
    best_c1, best_a1, best_e1 = 0.15, 1.0, None
    c = 0.01
    while c <= 0.60 + 1e-9:
        a = fit_a_through_origin(pairs, c)
        e = sse_yspace(pairs, a, c)
        if best_e1 is None or e < best_e1:
            best_e1, best_c1, best_a1 = e, c, a
        c += 0.01

    # Stage 2: pairs + clusters -> refine c locally, larger-S regime included
    pc = pairs + clusters if clusters else pairs
    best_c2, best_a2, best_e2 = best_c1, best_a1, None
    c = max(0.002, best_c1 - 0.05)
    hi = best_c1 + 0.05
    while c <= hi + 1e-9:
        a = fit_a_through_origin(pc, c)
        e = sse_yspace(pc, a, c)
        if best_e2 is None or e < best_e2:
            best_e2, best_c2, best_a2 = e, c, a
        c += 0.002

    # Stage 3: final light polish of the scale against ALL rows
    best_a3, best_e3 = best_a2, None
    scale = 0.85
    while scale <= 1.15 + 1e-9:
        a = best_a2 * scale
        e = sse_yspace(rows, a, best_c2)
        if best_e3 is None or e < best_e3:
            best_e3, best_a3 = e, a
        scale += 0.01

    a_final, c_final = best_a3, best_c2
    if a_final <= 0:
        a_final = 1e-3
    print("KERNEL %.6f / ( dist * dist + %.6f )" % (a_final, c_final))
    print("OUT v / ( 1 + S )")


if __name__ == "__main__":
    main()
