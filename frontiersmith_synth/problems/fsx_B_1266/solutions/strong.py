# TIER: strong
# Insight: equalize each sleeve's MARGINAL CONTRIBUTION TO TAIL LOSS, estimated from the
# STRESS-regime sample (not the calm one). At the current weights w, rank the visible
# stress scenarios by the PORTFOLIO's own loss (-stress_vis @ w) and take the worst
# fraction as the tail set -- this is where the stress-regime correlation structure
# actually enters: which scenarios are jointly worst depends on how the sleeves co-move
# under w, not on any single sleeve's own volatility. Because loss is linear in w, the
# tail-conditional CVaR decomposes exactly (Euler's identity) into per-sleeve
# contributions contribution_i = w_i * mean(-stress_r_i over the tail). Iteratively
# reweight (multiplicative fixed point) toward equal per-sleeve contribution, then
# project onto the per-sleeve caps and the cluster group cap. This is a genuine
# reformulation (risk parity in the tail, not mean-variance in the calm data, not a
# grid/random search) -- not "equal weight plus more iterations".
import sys
import numpy as np


def read_instance():
    toks = sys.stdin.read().split()
    pos = 0
    tid = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    cap = np.array([float(toks[pos + i]) for i in range(N)]); pos += N
    group_cap = float(toks[pos]); pos += 1
    K = int(toks[pos]); pos += 1
    cluster = [int(toks[pos + i]) for i in range(K)]; pos += K
    c_calm = int(toks[pos]); pos += 1
    pos += c_calm * N
    c_stress = int(toks[pos]); pos += 1
    stress = np.array([float(toks[pos + i]) for i in range(c_stress * N)]).reshape(c_stress, N)
    pos += c_stress * N
    return N, cap, group_cap, cluster, stress


def project(w0, cap, cluster_mask, group_cap, iters=300):
    w = np.clip(w0, 0, cap).astype(float)
    for _ in range(iters):
        csum = w[cluster_mask].sum()
        if csum > group_cap + 1e-9:
            w[cluster_mask] *= group_cap / csum
        w = np.minimum(w, cap)
        total = w.sum()
        if abs(total - 1.0) < 1e-10:
            break
        diff = 1.0 - total
        if diff > 0:
            slack = cap - w
            if w[cluster_mask].sum() >= group_cap - 1e-9:
                slack[cluster_mask] = 0.0
            ssum = slack.sum()
            if ssum <= 1e-12:
                break
            w += diff * slack / ssum
        else:
            pos = w > 1e-12
            psum = w[pos].sum()
            if psum <= 1e-12:
                break
            w[pos] += diff * w[pos] / psum
            w = np.maximum(w, 0.0)
    return w


def main():
    N, cap, group_cap, cluster, stress = read_instance()
    cluster_mask = np.zeros(N, dtype=bool)
    cluster_mask[cluster] = True

    w = project(np.ones(N) / N, cap, cluster_mask, group_cap)
    C = stress.shape[0]
    k_vis = max(2, round(0.4 * C))

    for _ in range(150):
        port_loss = -(stress @ w)
        tail_idx = np.argsort(port_loss)[-k_vis:]
        tail_loss_per_sleeve = -stress[tail_idx, :]
        contribution = w * tail_loss_per_sleeve.mean(axis=0)
        cvar_est = contribution.sum()
        target = max(cvar_est, 1e-8) / N
        contribution = np.maximum(contribution, 1e-8)
        w_new = w * np.sqrt(target / contribution)
        w_new = np.maximum(w_new, 1e-9)
        w_new = w_new / w_new.sum()
        w = project(w_new, cap, cluster_mask, group_cap)

    sys.stdout.write(" ".join("%.8f" % v for v in w) + "\n")


main()
