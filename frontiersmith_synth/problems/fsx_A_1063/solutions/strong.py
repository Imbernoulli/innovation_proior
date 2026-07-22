# TIER: strong
"""Exploit the resonant decoupling of the inverse Sturm-Liouville problem.

Key fact: writing sin^2(k*theta) = 1/2 - 1/2*cos(2k*theta), first-order
perturbation theory says the log-frequency shift of mode k caused by a mass
perturbation depends (almost) ONLY on the perturbation's own Fourier/DCT
component at spatial frequency 2k -- components at other frequencies are
(to leading order) ORTHOGONAL to mode k's sensitivity profile.  So instead
of moving mass to "the antinode of the mode I'm currently fixing" (which
is broadband and disturbs every other mode), we build the mass deviation
as a SUM of cosine components, one per target mode, cos(2k*pi*i/(n+1)),
each carrying its own DC-free amplitude a_k -- a resonant basis in which
the target modes become (nearly) INDEPENDENT control knobs.  We solve for
the amplitudes with a numerically-measured (finite-difference) Jacobian of
the TRUE nonlinear map (robust to boundary/finite-n deviations from the
idealized continuum formula), refine with a few damped Newton steps, and
finally use the smooth zero-mean SHAPE we found only to allocate money: a
1-D bisection on a uniform additive shift converts it into a nonnegative,
capped, exact-budget integer profile without disturbing that shape.
"""
import sys
import math
import numpy as np


def eigen_freq_ratios(n, m, r):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
    for i in range(n - 1):
        K[i, i + 1] = -1.0
        K[i + 1, i] = -1.0
    dinv = 1.0 / np.sqrt(m)
    Ksym = (dinv[:, None] * K) * dinv[None, :]
    ev = np.linalg.eigvalsh(Ksym)
    ev = np.clip(ev, 1e-12, None)
    ev = np.sort(ev)
    return np.sqrt(ev[:r])


def log_ratios(n, m, r):
    w = eigen_freq_ratios(n, m, r)
    return np.log(w / w[0])


def solve(n, B, CAP, r, targets):
    ks = list(range(2, r + 1))
    idx = np.arange(1, n + 1)
    basis = {}
    for k in ks:
        b = np.cos(2 * k * math.pi * idx / (n + 1))
        b = b - b.mean()  # zero-sum: never disturbs the total budget
        basis[k] = b

    m0 = 1.0 + B / n
    Lt = np.log(np.array(targets) / targets[0])
    m = np.full(n, float(m0))

    iters = 4
    damp = 0.8
    for _ in range(iters):
        mc = np.clip(m, 1.0, 1.0 + CAP)
        L_now = log_ratios(n, mc, r)
        err = Lt[1:] - L_now[1:]
        J = np.zeros((len(ks), len(ks)))
        h = 0.5
        for j, kp in enumerate(ks):
            mp = np.clip(mc + h * basis[kp], 1.0, 1.0 + CAP)
            Lp = log_ratios(n, mp, r)
            J[:, j] = (Lp[1:] - L_now[1:]) / h
        try:
            step = damp * np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            step = damp * np.linalg.lstsq(J, err, rcond=None)[0]
        for j, kp in enumerate(ks):
            m = m + step[j] * basis[kp]

    e_raw = m - 1.0  # unclipped resonant shape

    # Bisection on a uniform translation t so sum(clip(e_raw+t,0,CAP)) == B.
    lo = -CAP - float(np.max(np.abs(e_raw))) - 5.0
    hi = CAP + float(np.max(np.abs(e_raw))) + 5.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        s = np.clip(e_raw + mid, 0.0, CAP).sum()
        if s < B:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2.0
    e_cont = np.clip(e_raw + t, 0.0, CAP)

    e_int = np.floor(e_cont).astype(int)
    diff = B - int(e_int.sum())
    frac = e_cont - e_int
    order = np.argsort(-frac)
    j = 0
    guard = 0
    while diff > 0 and guard < 20 * n:
        i = order[j % n]
        if e_int[i] < CAP:
            e_int[i] += 1
            diff -= 1
        j += 1
        guard += 1
    i = 0
    while diff > 0 and i < n:
        if e_int[i] < CAP:
            take = min(diff, CAP - e_int[i])
            e_int[i] += take
            diff -= take
        i += 1
    i = 0
    while diff < 0 and i < n:
        if e_int[i] > 0:
            take = min(-diff, e_int[i])
            e_int[i] -= take
            diff += take
        i += 1

    return e_int


def main():
    data = sys.stdin.read().split()
    p = 0
    n = int(data[p]); p += 1
    B = int(data[p]); p += 1
    CAP = int(data[p]); p += 1
    r = int(data[p]); p += 1
    targets = []
    for _ in range(r):
        num = int(data[p]); p += 1
        den = int(data[p]); p += 1
        targets.append(num / den)

    e_int = solve(n, B, CAP, r, targets)
    print(" ".join(map(str, e_int.tolist())))


if __name__ == "__main__":
    main()
