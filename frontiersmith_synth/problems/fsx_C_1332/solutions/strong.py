# TIER: strong
"""The insight: matching the target profile at t=0 alone is the wrong
target -- the score is the mean match over the WHOLE four-hour trajectory,
and each ingredient's contribution decays on its own volatility-class
clock. So this solver:

  (1) TRAJECTORY-AWARE ALLOCATION -- treats each ingredient's usable
      "signal" as its descriptor vector broadcast across all 5 checkpoints
      through its OWN decay curve exp(-k_i*t) (a release schedule, not a
      single snapshot), and runs greedy matching pursuit against the
      concatenated (checkpoint x axis) target trajectory instead of just
      g(0). This is what lets it actually spend budget on a slow-decaying
      ingredient whose t=0 contribution looks unremarkable but whose t=4h
      contribution is the only thing that can still be there.

  (2) MASKING-AWARE LOCAL POLISH -- step (1) still assumes perceived
      intensity equals raw concentration (no masking). A short fixed
      number of coordinate-descent sweeps then perturbs each ingredient's
      concentration by a shrinking sequence of step sizes, accepting a
      move only if it reduces the EXACT nonlinear objective (the same
      decay+masking simulation the checker runs) -- this is what lets it
      correct for e.g. a base note being perceptually masked by a loud
      top note while both are still present.

Both stages respect the IFRA per-ingredient caps and the total-budget
constraint at every step, so the two mechanisms (decay schedule, masking)
and the regulatory limits all shape the final allocation together.
"""
import sys
import math

TOTAL_CAP = 1.0


def read_instance(text):
    toks = text.split()
    p = 0
    K = int(toks[p]); p += 1
    D = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    descs, ks, caps = [], [], []
    for _ in range(K):
        desc = [float(toks[p + a]) for a in range(D)]
        p += D
        k = float(toks[p]); p += 1
        cap = float(toks[p]); p += 1
        descs.append(desc); ks.append(k); caps.append(cap)
    mask = []
    for _ in range(K):
        row = [float(toks[p + j]) for j in range(K)]
        p += K
        mask.append(row)
    times = [float(toks[p + t]) for t in range(T)]
    p += T
    targets = []
    for _ in range(T):
        g = [float(toks[p + a]) for a in range(D)]
        p += D
        targets.append(g)
    return K, D, T, descs, ks, caps, mask, times, targets


def total_error(c0, descs, ks, mask, times, targets, K, D, T):
    tot = 0.0
    for ti in range(T):
        t = times[ti]
        raw = [c0[i] * math.exp(-ks[i] * t) for i in range(K)]
        perceived = [0.0] * K
        for i in range(K):
            denom = 1.0
            row = mask[i]
            for j in range(K):
                if j == i:
                    continue
                denom += row[j] * raw[j]
            perceived[i] = raw[i] / denom
        g = targets[ti]
        e = 0.0
        for a in range(D):
            s = 0.0
            for i in range(K):
                s += descs[i][a] * perceived[i]
            d = s - g[a]
            e += d * d
        tot += e
    return tot / T


def trajectory_matching_pursuit(descs, ks, caps, times, targets, K, D, T):
    """Greedy pursuit in the concatenated (checkpoint x axis) space, with
    each ingredient's feature broadcast through its OWN decay curve."""
    feat = []  # feat[i] = flattened D*T vector of desc_i[a]*exp(-k_i*t)
    for i in range(K):
        v = []
        for ti in range(T):
            decay = math.exp(-ks[i] * times[ti])
            for a in range(D):
                v.append(descs[i][a] * decay)
        feat.append(v)
    target_flat = []
    for ti in range(T):
        for a in range(D):
            target_flat.append(targets[ti][a])
    L = D * T

    alloc = [0.0] * K
    budget = TOTAL_CAP
    residual = list(target_flat)

    MAX_PASSES = 4 * K + 4
    for _ in range(MAX_PASSES):
        if budget <= 1e-9:
            break
        best_i, best_x, best_gain = -1, 0.0, 0.0
        for i in range(K):
            room = caps[i] - alloc[i]
            if room <= 1e-9:
                continue
            room = min(room, budget)
            fv = feat[i]
            dot = sum(residual[q] * fv[q] for q in range(L))
            nrm = sum(fv[q] * fv[q] for q in range(L))
            if nrm <= 1e-12:
                continue
            x_star = dot / nrm
            x = min(max(x_star, 0.0), room)
            if x <= 1e-9:
                continue
            gain = 2.0 * x * dot - x * x * nrm
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
                best_x = x
        if best_i < 0:
            break
        fv = feat[best_i]
        for q in range(L):
            residual[q] -= best_x * fv[q]
        alloc[best_i] += best_x
        budget -= best_x

    return alloc


def local_polish(alloc, descs, ks, caps, mask, times, targets, K, D, T):
    c = list(alloc)
    best_err = total_error(c, descs, ks, mask, times, targets, K, D, T)
    steps = [0.08, 0.05, 0.03, 0.015, 0.008, 0.004, 0.002]
    for _sweep in range(4):
        improved_any = False
        for i in range(K):
            for step in steps:
                for sign in (1.0, -1.0):
                    delta = sign * step
                    new_ci = c[i] + delta
                    if new_ci < 0.0 or new_ci > caps[i] + 1e-12:
                        continue
                    others = sum(c) - c[i]
                    if others + new_ci > TOTAL_CAP + 1e-9:
                        continue
                    trial = list(c)
                    trial[i] = new_ci
                    err = total_error(trial, descs, ks, mask, times, targets, K, D, T)
                    if err < best_err - 1e-12:
                        c = trial
                        best_err = err
                        improved_any = True
        if not improved_any:
            break
    return c


def main():
    text = sys.stdin.read()
    K, D, T, descs, ks, caps, mask, times, targets = read_instance(text)

    alloc = trajectory_matching_pursuit(descs, ks, caps, times, targets, K, D, T)
    alloc = local_polish(alloc, descs, ks, caps, mask, times, targets, K, D, T)

    used = [(i + 1, alloc[i]) for i in range(K) if alloc[i] > 1e-9]
    if not used:
        used = [(1, min(caps[0], TOTAL_CAP))]

    out = [str(len(used))]
    for idx, c in used:
        out.append("%d %.6f" % (idx, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
