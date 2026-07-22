import sys, random

W = 100          # peer weight: each already-adopted upstream neighbor adds W to activation
THETA_EAGER = 1  # eager households (almost ready; uniform subsidy already flips them)
THETA_ISO   = 25 # isolated households, no useful neighbors
HEAD_THETA  = 150
DELTA       = 10 # near-threshold top-up needed at each corridor step
INTERIOR_THETA = W + DELTA   # = 110

def corridor_full_cost(L):   # cost for one whole corridor via continuation (skip head)
    return 110 + 10 * (L - 2)          # enter interior1 at 110, then (L-2) steps at 10
def corridor_partial_cost(rem):        # rem>=1 corridor adopters
    return 110 + 10 * (rem - 1)

def budget_for_target(E, I, L, target):
    """Budget so the continuation strategy reaches exactly `target` adopters:
       eager(1) then isolated(25) then corridors (110 entry + 10/step)."""
    c = E * THETA_EAGER + I * THETA_ISO
    rem = target - E - I
    if rem <= 0:
        return max(c, 1)
    full = rem // (L - 1)
    part = rem % (L - 1)
    c += full * corridor_full_cost(L)
    if part > 0:
        c += corridor_partial_cost(part)
    return c

def main():
    t = int(sys.argv[1])
    rng = random.Random(90580 + t)
    E = 20 + 4 * t
    I = 15 + 3 * t
    L = 20 + 3 * t
    K = 20                      # plenty of corridors (continuation uses < 8)
    Rz = 30 + 10 * t            # resistant households (dilution + decoys)

    theta = []; r = []; edges = []
    # eager
    for _ in range(E):
        theta.append(THETA_EAGER); r.append(1)
    # isolated
    for _ in range(I):
        theta.append(THETA_ISO); r.append(1)
    # corridors (directed head -> ... -> tail)
    for _ in range(K):
        base = len(theta)
        for j in range(L):
            theta.append(HEAD_THETA if j == 0 else INTERIOR_THETA); r.append(1)
        for j in range(L - 1):
            edges.append((base + j, base + j + 1))
    # resistant
    for _ in range(Rz):
        theta.append(10 ** 9); r.append(1)

    n = len(theta)

    # budget: continuation strategy should land near 6.8x the uniform baseline (=E)
    target = round(6.8 * E)
    B = budget_for_target(E, I, L, target)

    # shuffle node labels so structure is not readable from index order
    perm = list(range(n))
    rng.shuffle(perm)
    old2new = [0] * n
    for new, old in enumerate(perm):
        old2new[old] = new
    ntheta = [0] * n; nr = [0] * n
    for old in range(n):
        ntheta[old2new[old]] = theta[old]
        nr[old2new[old]] = r[old]
    nedges = [(old2new[u], old2new[v]) for (u, v) in edges]
    rng.shuffle(nedges)

    out = []
    out.append("%d %d %d %d" % (n, len(nedges), B, W))
    for i in range(n):
        out.append("%d %d" % (ntheta[i], nr[i]))
    for (u, v) in nedges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
