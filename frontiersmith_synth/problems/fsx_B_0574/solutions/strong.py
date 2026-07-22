# TIER: strong
# Insight: the optimal mass split is dictated by the DOWNSTREAM jettison-and-loss
# schedule, not by any single stage's local efficiency.  We optimise the exact
# objective (Tsiolkovsky cascade minus the burn-duration * L_i loss) jointly over
# the integer engine counts and the continuous propellant split by loss-aware
# coordinate ascent, seeded from a geometric staging.  High-L stages are pushed
# toward shorter burns (more engines / less propellant); the freed mass is
# reallocated where the downstream cascade pays for it.
import sys, math


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    S = int(float(next(it)))
    P = float(next(it)); M_total = float(next(it)); kappa = float(next(it))
    m_e = float(next(it)); T = float(next(it)); v_e = float(next(it))
    g = float(next(it)); E_max = int(float(next(it)))
    L = [float(next(it)) for _ in range(S)]
    return dict(S=S, P=P, M_total=M_total, kappa=kappa, m_e=m_e, T=T, v_e=v_e,
               g=g, E_max=E_max, L=L)


def velocity(q, n, p):
    S, P, m_e, kappa = q["S"], q["P"], q["m_e"], q["kappa"]
    T, v_e, g, L = q["T"], q["v_e"], q["g"], q["L"]
    suffix = 0.0
    parts = [0.0] * (S + 1)
    for i in range(S - 1, -1, -1):
        suffix += n[i] * m_e + (1.0 + kappa) * p[i]
        parts[i] = suffix
    V = 0.0
    for i in range(S):
        m_start = P + parts[i]
        m_end = m_start - p[i]
        if m_end <= 0.0 or m_start <= 0.0:
            return float("-inf")
        V += v_e * math.log(m_start / m_end) - L[i] * g * (p[i] * v_e / (n[i] * T))
    return V


def avail_prop(q, n):
    return (q["M_total"] - q["P"] - sum(ni * q["m_e"] for ni in n)) / (1.0 + q["kappa"])


def main():
    q = read_instance()
    S, m_e, E_max = q["S"], q["m_e"], q["E_max"]

    # ---- seed: geometric split, modest engine count ----
    n = [min(E_max, 3)] * S
    avail = avail_prop(q, n)
    if avail <= 0:
        n = [1] * S
        avail = avail_prop(q, n)
    R = (q["M_total"] / q["P"]) ** (1.0 / S)
    w = [max(1e-6, R ** (S - i)) for i in range(S)]
    sw = sum(w)
    p = [avail * wi / sw for wi in w]

    def renorm(p, target):
        s = sum(p)
        if s <= 0:
            return [target / len(p)] * len(p)
        return [max(0.0, x * target / s) for x in p]

    best = velocity(q, n, p)

    for _ in range(8):
        # (A) engine-count coordinate ascent (loss-aware: high L wants more engines)
        for i in range(S):
            cur_n = n[i]
            cur_best_v, cur_best_n, cur_best_p = best, cur_n, p
            for cand in range(1, E_max + 1):
                n2 = list(n); n2[i] = cand
                a2 = avail_prop(q, n2)
                if a2 <= 0:
                    continue
                p2 = renorm(p, a2)
                v2 = velocity(q, n2, p2)
                if v2 > cur_best_v + 1e-9:
                    cur_best_v, cur_best_n, cur_best_p = v2, cand, p2
            if cur_best_n != cur_n:
                n[i] = cur_best_n
                p = cur_best_p
                best = cur_best_v

        # (B) propellant mass-transfer local search with shrinking steps
        avail = avail_prop(q, n)
        p = renorm(p, avail)
        best = velocity(q, n, p)
        step = avail * 0.25
        for _step_round in range(50):
            improved = False
            for i in range(S):
                for j in range(S):
                    if i == j:
                        continue
                    ds = step
                    if p[i] < ds:
                        ds = p[i]
                    if ds <= 1e-9:
                        continue
                    p2 = list(p)
                    p2[i] -= ds
                    p2[j] += ds
                    v2 = velocity(q, n, p2)
                    if v2 > best + 1e-7:
                        p = p2
                        best = v2
                        improved = True
            if not improved:
                step *= 0.5
                if step < avail * 1e-5:
                    break

    out = ["%d %.6f" % (n[i], p[i]) for i in range(S)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
