# TIER: strong
# Sample-average approximation (SAA) + coordinate descent.  The candidate cannot see
# the hidden survey-day scenarios, so it draws its OWN M scenarios from the declared
# distributions and minimizes the SAME composite objective (holding + shortage +
# transfer + service penalty) on them, subject to the budget:
#   * start from the per-site critical fractile;
#   * line-search each local stock q_i over a small multiplicative grid (and down to
#     the service-level floor), keeping any change that lowers the empirical cost and
#     stays within budget;
#   * grow the vessel reserve q0, exploiting risk pooling (one shared buffer covers
#     whichever site happens to run short), which the single-echelon greedy ignores.
# Because it is fit on independent samples and evaluated on held-out scenarios, it
# generalizes and beats the closed-form fractile, while the loose normalization keeps
# it well below 1.0.
import sys, json, math


def inv_norm(p):
    if p <= 0:
        return -1e9
    if p >= 1:
        return 1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def rng(seed):
    state = (int(seed) * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u


def normals(u, k):
    out = []
    while len(out) < k:
        u1 = u()
        u2 = u()
        if u1 < 1e-12:
            u1 = 1e-12
        rr = math.sqrt(-2.0 * math.log(u1))
        out.append(rr * math.cos(2 * math.pi * u2))
        out.append(rr * math.sin(2 * math.pi * u2))
    return out[:k]


inst = json.load(sys.stdin)
sites = inst["sites"]
cen = inst["central"]
N = inst["n_sites"]
B = inst["budget"]
tau = inst["tau"]
lam = inst["lam"]

M = 150
u = rng(20260701)
D = []
for s in sites:
    z = normals(u, M)
    col = []
    for k in range(M):
        if s["dist"] == "normal":
            dv = s["mean"] + s["std"] * z[k]
        elif s["dist"] == "lognormal":
            cv = s["std"] / s["mean"]
            sig = math.sqrt(math.log(1 + cv * cv))
            mu = math.log(s["mean"]) - sig * sig / 2
            dv = math.exp(mu + sig * z[k])
        else:
            dv = (s["mean"] * 0.4 + s["std"] * z[k] * 0.3) if u() < 0.5 \
                else (s["mean"] * 1.6 + s["std"] * z[k] * 0.3)
        col.append(max(0.0, dv))
    D.append(col)

order = sorted(range(N), key=lambda i: (sites[i]["p"] - cen["t"]), reverse=True)


def emp(q, q0):
    total = 0.0
    stockout = [0] * N
    for k in range(M):
        rem = q0
        sc = 0.0
        short = [0.0] * N
        left = [0.0] * N
        for i in range(N):
            d = D[i][k]
            if q[i] >= d:
                left[i] = q[i] - d
            else:
                short[i] = d - q[i]
                stockout[i] += 1
        for i in order:
            if short[i] > 0 and sites[i]["p"] > cen["t"] and rem > 0:
                cov = short[i] if short[i] < rem else rem
                rem -= cov
                sc += cen["t"] * cov
                short[i] -= cov
        for i in range(N):
            sc += sites[i]["h"] * left[i] + sites[i]["p"] * short[i]
        sc += cen["h0"] * rem
        total += sc
    avg = total / M
    serv = 0.0
    for i in range(N):
        phat = 1.0 - stockout[i] / M
        serv += lam * max(0.0, tau - phat) * sites[i]["p"] * sites[i]["mean"]
    return avg + serv


def spend(q, q0):
    return sum(sites[i]["c"] * q[i] for i in range(N)) + cen["c0"] * q0


def feas(q, q0):
    return spend(q, q0) <= B * (1 + 1e-9)


ztau = inv_norm(tau)
floor = [max(0.0, sites[i]["mean"] + ztau * sites[i]["std"]) for i in range(N)]
q = [max(0.0, sites[i]["mean"] + inv_norm(sites[i]["p"] / (sites[i]["p"] + sites[i]["h"])) * sites[i]["std"])
     for i in range(N)]
q0 = 0.0
if not feas(q, q0):
    sc = B / max(spend(q, 0.0), 1e-9)
    q = [x * sc for x in q]

best = emp(q, q0)
for _ in range(4):
    for i in range(N):
        base = q[i]
        for cand in (base * 0.7, base * 0.85, base * 0.95, base * 1.1, base * 1.25, floor[i]):
            q[i] = cand
            if feas(q, q0):
                o = emp(q, q0)
                if o < best - 1e-9:
                    best = o
                else:
                    q[i] = base
            else:
                q[i] = base
    for step in (B * 0.02, B * 0.05, B * 0.1):
        cand0 = q0 + step / cen["c0"]
        if feas(q, cand0):
            o = emp(q, cand0)
            if o < best - 1e-9:
                best = o
                q0 = cand0

print(json.dumps({"q": q, "q0": q0}))
