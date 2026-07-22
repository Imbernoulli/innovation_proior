#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE pigment-metamerism-matching instance to stdout.

Instance layout (all deterministic given testId; seeded via testId only):

  N M K
  xbar(1..N)
  ybar(1..N)
  zbar(1..N)
  L_1(1..N)
  ...
  L_K(1..N)
  Rtarget(1..N)
  cost_weight
  K_1(1..N)
  S_1(1..N)
  ...
  K_M(1..N)
  S_M(1..N)
  cost_1 ... cost_M

N = number of spectral bands, M = number of candidate pigments, K = number of
illuminants. xbar/ybar/zbar are fixed synthetic colour-matching functions.
L_j are illuminant power spectra. Rtarget is the reflectance spectrum to
match. cost_weight scales the "pigments used" penalty. Each pigment i has a
Kubelka-Munk absorption spectrum K_i and scattering spectrum S_i, and a
per-unit cost cost_i.

Trap construction: illuminant L_1 has near-zero power on the last three
("blind") bands, while L_2 and L_3 have substantial power there (L_3 is a
narrow-band illuminant that spikes exactly on those bands). Two cheap
"decoy" pigments are planted whose reflectance, taken alone, reproduces the
target EXACTLY on the non-blind bands but diverges sharply on the blind
bands -- a perfect (and cheap) metameric match under L_1 alone that fails
badly under L_2/L_3. A disjoint set of "true" pigments is planted whose
KM-forward mixture reproduces the (lightly perturbed) target across ALL
bands, so matching the full spectral shape is illuminant-invariant.
"""
import sys
import math
import random


def gauss(x, c, s):
    return math.exp(-0.5 * ((x - c) / s) ** 2)


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_curve(rng, n, base, amp_lo, amp_hi, n_bumps, lo, hi):
    centers = [rng.uniform(0, n - 1) for _ in range(n_bumps)]
    sigmas = [rng.uniform(1.3, 3.2) for _ in range(n_bumps)]
    amps = [rng.uniform(amp_lo, amp_hi) for _ in range(n_bumps)]
    out = []
    for b in range(n):
        v = base
        for c, s, a in zip(centers, sigmas, amps):
            v += a * gauss(b, c, s)
        out.append(clip(v, lo, hi))
    return out


def km_reflectance(ratio):
    # single-constant Kubelka-Munk, infinite-thickness reflectance
    r = max(0.0, ratio)
    return 1.0 + r - math.sqrt(r * r + 2.0 * r)


def km_mix_reflectance(weights, Ks, Ss, n):
    R = []
    for b in range(n):
        kmix = sum(w * K[b] for w, K in zip(weights, Ks))
        smix = sum(w * S[b] for w, S in zip(weights, Ss))
        smix = max(smix, 1e-6)
        R.append(km_reflectance(kmix / smix))
    return R


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20260 + 97 * testId)

    N = 12
    BLIND = [9, 10, 11]  # bands illuminant L1 barely sees

    M = 6 + testId               # 7 .. 16 pigments
    n_decoy = 2
    n_true = 3 if testId % 2 == 0 else 4
    n_filler = M - n_decoy - n_true
    if n_filler < 1:
        n_filler = 1
        M = n_decoy + n_true + n_filler

    # ---- fixed synthetic colour matching functions ----
    xbar = [0.9 * gauss(b, 2.0, 1.2) + 0.35 * gauss(b, 9.0, 1.5) for b in range(N)]
    ybar = [1.0 * gauss(b, 5.5, 2.0) for b in range(N)]
    zbar = [1.15 * gauss(b, 1.0, 1.0) for b in range(N)]

    # ---- illuminants ----
    L1 = []
    for b in range(N):
        if b in BLIND:
            L1.append(clip(0.04 + rng.uniform(-0.01, 0.02), 0.01, 0.2))
        else:
            L1.append(clip(0.85 + 0.4 * gauss(b, 4.0, 4.0) + rng.uniform(-0.05, 0.05), 0.2, 1.6))

    L2 = []
    for b in range(N):
        v = 0.9 + 0.35 * math.sin(0.55 * b + 1.0) + rng.uniform(-0.05, 0.05)
        L2.append(clip(v, 0.5, 1.4))

    L3 = [0.28 + rng.uniform(-0.03, 0.03) for _ in range(N)]
    spike_vals = [2.6, 2.1, 2.9]
    for k, b in enumerate(BLIND):
        L3[b] = clip(spike_vals[k] + rng.uniform(-0.1, 0.1), 1.8, 3.2)

    illuminants = [L1, L2, L3]
    K_ill = len(illuminants)

    # ---- "true" pigments: their KM-forward mixture will define the target ----
    true_ratio_curves = []
    true_S = []
    true_K = []
    for _ in range(n_true):
        Sc = smooth_curve(rng, N, 0.45, 0.15, 0.35, 2, 0.2, 1.0)
        ratio_c = smooth_curve(rng, N, 0.25, 0.4, 1.3, 2, 0.05, 3.0)
        Kc = [ratio_c[b] * Sc[b] for b in range(N)]
        true_S.append(Sc)
        true_K.append(Kc)
        true_ratio_curves.append(ratio_c)

    raw_w = [rng.uniform(0.4, 1.6) for _ in range(n_true)]
    tot = sum(raw_w)
    true_weights = [w / tot for w in raw_w]

    base_target = km_mix_reflectance(true_weights, true_K, true_S, N)
    # small deterministic smooth perturbation so the target is not EXACTLY
    # reachable (keeps headroom; also final decoy ratios derive from the
    # perturbed target so the trap remains exact on the non-blind bands)
    noise = smooth_curve(rng, N, 0.0, -0.02, 0.02, 3, -0.03, 0.03)
    Rtarget = [clip(base_target[b] + noise[b], 0.03, 0.97) for b in range(N)]

    def inv_km(Rv):
        Rv = clip(Rv, 1e-4, 1 - 1e-4)
        return (1.0 - Rv) ** 2 / (2.0 * Rv)

    target_ratio = [inv_km(Rtarget[b]) for b in range(N)]

    # ---- decoy pigments: S == 1, K == target_ratio on non-blind bands,
    # deliberately wrong (mirrored / rescaled) on the blind bands ----
    decoy_S = []
    decoy_K = []
    for d in range(n_decoy):
        Sc = [1.0] * N
        Kc = []
        for b in range(N):
            if b in BLIND:
                # deliberately wrong value at a blind band: two different
                # wrong directions for the two decoys
                far = target_ratio[b]
                if d == 0:
                    wrong = clip(far * 3.2 + 0.6, 0.05, 4.5)
                else:
                    wrong = clip(far * 0.15, 0.01, 4.5)
                Kc.append(wrong)
            else:
                Kc.append(target_ratio[b])
        decoy_S.append(Sc)
        decoy_K.append(Kc)

    # ---- filler pigments: unrelated smooth palette clutter ----
    filler_S = []
    filler_K = []
    for _ in range(n_filler):
        Sc = smooth_curve(rng, N, 0.4, 0.1, 0.4, 2, 0.15, 1.0)
        ratio_c = smooth_curve(rng, N, 0.3, 0.2, 1.6, 2, 0.05, 3.5)
        Kc = [ratio_c[b] * Sc[b] for b in range(N)]
        filler_S.append(Sc)
        filler_K.append(Kc)

    # ---- assemble palette & costs, then SHUFFLE the order together so no
    # solution can exploit "true pigments come first" positional structure
    all_K = true_K + decoy_K + filler_K
    all_S = true_S + decoy_S + filler_S
    costs = []
    for _ in range(n_true):
        costs.append(round(rng.uniform(1.5, 2.5), 3))
    for _ in range(n_decoy):
        costs.append(1.0)  # decoys are the cheapest option -> extra bait
    for _ in range(n_filler):
        costs.append(round(rng.uniform(1.0, 3.0), 3))

    order = list(range(len(all_K)))
    rng.shuffle(order)
    all_K = [all_K[i] for i in order]
    all_S = [all_S[i] for i in order]
    costs = [costs[i] for i in order]

    Mtot = len(all_K)
    cost_weight = round(0.35 + 0.03 * (testId % 5), 3)

    # ---- emit ----
    out = []
    out.append(f"{N} {Mtot} {K_ill}")
    out.append(" ".join(f"{v:.6f}" for v in xbar))
    out.append(" ".join(f"{v:.6f}" for v in ybar))
    out.append(" ".join(f"{v:.6f}" for v in zbar))
    for L in illuminants:
        out.append(" ".join(f"{v:.6f}" for v in L))
    out.append(" ".join(f"{v:.6f}" for v in Rtarget))
    out.append(f"{cost_weight:.6f}")
    for Kc, Sc in zip(all_K, all_S):
        out.append(" ".join(f"{v:.6f}" for v in Kc))
        out.append(" ".join(f"{v:.6f}" for v in Sc))
    out.append(" ".join(f"{v:.6f}" for v in costs))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
