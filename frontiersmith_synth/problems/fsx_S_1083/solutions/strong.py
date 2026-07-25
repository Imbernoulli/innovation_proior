# TIER: strong
# Rank-structure insight about the ENSEMBLE of bases: form the ensemble Gram
# G = sum_k B_k B_k^T. Its eigen-directions with eigenvalue ~K span the SHARED
# subspace (every family lies there); eigenvalue ~1 directions are
# family-specific. Split the probe budget into
#   (a) a structured part: eigen-directions of G chosen by a greedy
#       worst-family rule -- repeatedly add the direction that most increases
#       the covered mass of the currently least-covered family (so the shared
#       subspace is bought once and small family-specific parts top up), and
#   (b) a small flat random part (~15%) that keeps the measured dictionary
#       well-conditioned and the per-measurement SNR high.
# Structured directions are quantized to integer probes with a per-direction
# scale search that minimizes quantization error under the entry bound.
import sys
import numpy as np

RAND_FRAC = 0.15
SEED = 424242


def quant_best(w, pmax):
    """Best integer approximation of direction w: search scale a so that
    round(a*w) stays within [-pmax, pmax] and minimizes relative error."""
    mx = float(np.max(np.abs(w)))
    best, besterr = None, 1e18
    for a in np.linspace(0.5, pmax / mx, 80):
        q = np.rint(a * w)
        if np.max(np.abs(q)) > pmax or np.all(q == 0):
            continue
        err = float(np.linalg.norm(q - a * w) / (np.linalg.norm(a * w) + 1e-12))
        if err < besterr:
            besterr, best = err, q
    return best


def main():
    data = sys.stdin.read().split()
    pos = 0
    n, m, K, s, pmax = (int(data[i]) for i in range(5))
    pos = 5
    bases = []
    for _ in range(K):
        d = int(data[pos]); pos += 1
        B = np.array([float(x) for x in data[pos:pos + n * d]]).reshape(n, d)
        pos += n * d
        bases.append(B)

    G = np.zeros((n, n))
    for B in bases:
        G += B @ B.T
    lam, W = np.linalg.eigh(G)
    idx = np.argsort(-lam)
    lam = lam[idx]
    W = W[:, idx]
    cand = [i for i in range(n) if lam[i] > 1e-8]
    L = len(cand)

    rng = np.random.default_rng(SEED)
    m_rand = int(round(m * RAND_FRAC))
    m_struct = m - m_rand

    if L <= m_struct:
        sel = list(cand)
    else:
        # greedy worst-family (submodular) selection over eigen-directions
        Bproj = [B.T @ W[:, cand] for B in bases]  # d_k x L
        dims = [B.shape[1] for B in bases]
        mass = [0.0] * K
        sel = []
        for _ in range(m_struct):
            worst = int(np.argmin([mass[k] / dims[k] for k in range(K)]))
            best_j, best_val = None, -1.0
            for j in range(L):
                if j in sel:
                    continue
                gain = float(np.sum(Bproj[worst][:, j] ** 2))
                if gain > best_val:
                    best_val, best_j = gain, j
            if best_j is None:
                break
            sel.append(best_j)
            for k in range(K):
                mass[k] += float(np.sum(Bproj[k][:, best_j] ** 2))
        sel = [cand[j] for j in sel]

    probes = [quant_best(W[:, i], pmax) for i in sel]
    while len(probes) < m:
        probes.append(rng.choice([-pmax, pmax], size=n).astype(float))
    probes = probes[:m]

    sys.stdout.write("\n".join(" ".join(str(int(v)) for v in row) for row in probes) + "\n")


main()
