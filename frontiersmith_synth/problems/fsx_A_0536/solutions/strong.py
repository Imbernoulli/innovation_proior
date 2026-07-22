# TIER: strong
# INSIGHT: effective resistances are globally COUPLED through the Laplacian
# pseudoinverse -- you cannot set a wire to 1/target and read that target back,
# because every other wire (and the shared backbone) already lowers it. So treat the
# ENTIRE conductance vector as one design and optimize it against the joint response.
# The gradient is analytic and MONOTONE: for edge e=(a,b) and pair k=(i,j),
#   dR_k/dc_e = -(Lp[a,i]-Lp[a,j]-Lp[b,i]+Lp[b,j])^2  <= 0,
# so strengthening any wire lowers every resistance. We descend the weighted clipped
# relative error over log-conductances with backtracking line search, tuning the
# SHARED backbone wires as well as the per-pair wires, and keep the best iterate.
import sys
import numpy as np

RELCAP = 3.0


def build_L(n, U, V, C):
    L = np.zeros((n, n), dtype=float)
    np.add.at(L, (U, U), C)
    np.add.at(L, (V, V), C)
    np.add.at(L, (U, V), -C)
    np.add.at(L, (V, U), -C)
    return L


def main():
    tok = sys.stdin.read().split()
    pos = 0
    n = int(tok[pos]); pos += 1
    m = int(tok[pos]); pos += 1
    wmax = float(tok[pos]); pos += 1
    P = int(tok[pos]); pos += 1
    I = np.empty(P, dtype=int); J = np.empty(P, dtype=int)
    T = np.empty(P); W = np.empty(P)
    for k in range(P):
        a = int(tok[pos]); b = int(tok[pos + 1])
        T[k] = float(tok[pos + 2]); W[k] = float(tok[pos + 3]); pos += 4
        if a > b:
            a, b = b, a
        I[k] = a; J[k] = b

    active = sorted(set(I.tolist()) | set(J.tolist()))

    eset = {}
    edges = []          # (u, v)

    def add(u, v):
        a, b = (u, v) if u < v else (v, u)
        if a == b or (a, b) in eset:
            return False
        eset[(a, b)] = len(edges)
        edges.append((a, b))
        return True

    # backbone spanning path + a direct wire per pair (as budget allows)
    for k in range(len(active) - 1):
        add(active[k], active[k + 1])
    tgt_of = {}
    order = sorted(range(P), key=lambda k: -W[k])
    for k in order:
        if len(edges) >= m:
            break
        if add(int(I[k]), int(J[k])):
            tgt_of[(int(I[k]), int(J[k]))] = k

    E = len(edges)
    U = np.array([e[0] for e in edges]); Vv = np.array([e[1] for e in edges])
    # initial conductances: 1/target for pair wires, 1.0 for backbone, capped
    C = np.ones(E)
    for ei, (a, b) in enumerate(edges):
        if (a, b) in tgt_of:
            C[ei] = min(wmax, 1.0 / T[tgt_of[(a, b)]])
    C = np.clip(C, 1e-4, wmax)
    Wsum = float(W.sum())

    def resistances(Cv):
        L = build_L(n, U, Vv, Cv)
        Lp = np.linalg.pinv(L, hermitian=True)
        R = Lp[I, I] + Lp[J, J] - 2.0 * Lp[I, J]
        return R, Lp

    def loss(R):
        rel = (R - T) / T
        rel = np.clip(rel, -RELCAP, RELCAP)
        return float((W * rel * rel).sum() / Wsum)

    R, Lp = resistances(C)
    bestC = C.copy(); bestF = loss(R)

    lr = 0.5
    for it in range(45):
        R, Lp = resistances(C)
        rel = (R - T) / T
        active_mask = (np.abs(rel) < RELCAP).astype(float)  # clipped pairs have 0 grad
        # coef_k = d(loss)/dR_k = (2 w_k / Wsum) * rel_k / t_k   (only unclipped)
        coef = (2.0 * W / Wsum) * rel / T * active_mask
        # D[e,k] = Lp[a,i]-Lp[a,j]-Lp[b,i]+Lp[b,j];  dR_k/dc_e = -D^2
        A_ = Lp[U][:, I]; B_ = Lp[U][:, J]; Cc_ = Lp[Vv][:, I]; Dd_ = Lp[Vv][:, J]
        D = A_ - B_ - Cc_ + Dd_            # (E,P)
        grad_c = -(D * D) @ coef           # d loss / d c_e   (E,)
        glog = grad_c * C                  # gradient wrt log c
        gnorm = np.linalg.norm(glog)
        if gnorm < 1e-12:
            break
        step = lr / max(1.0, gnorm)
        improved = False
        for _ in range(6):
            Cn = np.clip(C * np.exp(-step * glog), 1e-4, wmax)
            Rn, _ = resistances(Cn)
            Fn = loss(Rn)
            if Fn < bestF - 1e-12:
                bestF = Fn; bestC = Cn.copy()
                C = Cn; improved = True
                break
            step *= 0.4
        if not improved:
            lr *= 0.5
            if lr < 1e-3:
                break

    out = [str(E)]
    for (e, c) in zip(edges, bestC):
        cc = min(wmax, max(1e-6, float(c)))
        out.append("%d %d %.6f" % (e[0], e[1], cc))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
