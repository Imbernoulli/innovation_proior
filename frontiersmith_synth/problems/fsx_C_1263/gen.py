#!/usr/bin/env python3
"""
gen.py <testId> -- optical WDM link-budget instance generator (family: optical-link-budget).
Deterministic: all randomness seeded ONLY from testId.

Physical model (abstracted, unit-free)
---------------------------------------
A fibre link carries C wavelength channels through S amplified spans. Each span s
contributes an additive amplified-spontaneous-emission (ASE) noise term ase[s]; the
total ASE noise floor seen by every channel is N_ASE = sum(ase[1..S]) -- more spans
(longer haul) accumulate strictly more noise (mechanism: amplifier-noise-accumulation).

Each channel c also suffers fibre-nonlinearity noise that grows with its OWN launch
power P_c and with its NEIGHBOURS' launch powers (cross-phase modulation / four-wave
mixing between wavelengths sharing the same fibre):

    Q_c(P)   = sum_{c' != c} kappa[c][c'] * P_{c'}^2
    NLI_c(P) = eta_c * P_c^3 + Q_c(P) * P_c^2
    SNR_c(P) = P_c / (N_ASE + NLI_c(P))                (0 if P_c == 0)

Raising P_c alone raises SNR_c roughly linearly at first (ASE-limited) but the
eta_c*P_c^3 self-nonlinearity term eventually dominates and SNR_c FALLS -- an interior
optimum in P_c (mechanism: launch-power-vs-nonlinearity). The kappa cross term means a
channel's optimum also depends on what its neighbours are doing.

A channel is assigned a modulation tier m (1..K, or 0 = silent/off) from a fixed table
(bps[k], req_snr[k]) shared by every channel; tier k is usable only if SNR_c >= req_snr[k]
(mechanism: modulation-order-choice -- higher bps needs higher SNR, so the *reachable*
tier is coupled to the chosen power, not chosen independently of it).

Difficulty ladder / trap design
--------------------------------
testId controls channel count C (3..12, growing with testId) and the per-instance
"haul length" regime:
  SHORT_IDS  (favourable to naive max-power): few spans, and Pmax_c is deliberately
    kept BELOW each channel's isolated SNR-maximising power peak0_c = (N_ASE/(2 eta_c))**(1/3)
    -- so pushing every channel to its ceiling is close to correct there.
  TRAP_IDS (the majority, >=3 of 10 by construction): many spans (large N_ASE) AND
    Pmax_c set to 2.5x-4.5x peak0_c -- "use all the power you're allowed" drives every
    channel FAR past its own SNR peak into the nonlinear-noise-dominated regime, and
    because every channel does this simultaneously the kappa cross term compounds the
    damage. Reaching near-optimal throughput here requires backing power off toward
    (and, per channel, tightly around) the *joint* SNR peak instead of the ceiling.

Every ase[s] and eta_c are positive, so peak0_c > 0 and N_ASE > 0 always; Pmax_c is
always set to at least a small positive floor so every instance admits a nonzero-rate
feasible schedule (the checker's own low-power reference construction).
"""
import random
import sys

SHORT_IDS = {1, 2, 6}   # favour high power (max-power close to optimal)
# all other ids in 1..10 are TRAP ids (long-haul / power-far-above-peak)

BPS = [1, 2, 4, 6]
REQ_SNR = [1.0, 2.5, 6.0, 15.0]
K = len(BPS)


def peak0(n_ase, eta):
    return (n_ase / (2.0 * eta)) ** (1.0 / 3.0)


def gen(test_id: int):
    rnd = random.Random(700001 + 131 * test_id)

    trap = test_id not in SHORT_IDS
    C = 3 + (test_id - 1)          # 3..12
    C = min(C, 12)

    S = rnd.randint(1, 4) if not trap else rnd.randint(9, 24 + min(test_id, 6))
    S = max(1, min(S, 30))

    ase = [rnd.randint(3, 5) for _ in range(S)]
    n_ase = float(sum(ase))

    eta = [rnd.uniform(0.5e-6, 2.0e-6) for _ in range(C)]
    peak0_c = [peak0(n_ase, eta[c]) for c in range(C)]

    if trap:
        factor = [rnd.uniform(1.5, 2.4) for _ in range(C)]
    else:
        factor = [rnd.uniform(0.55, 0.85) for _ in range(C)]
    pmax = [round(factor[c] * peak0_c[c], 3) for c in range(C)]
    pmax = [max(p, 1.0) for p in pmax]  # tiny positive floor, never 0

    # Cross-channel coupling: kappa[c][c'] = zeta_c * eta_c / peak0_c / (1 + |c-c'|).
    # Dividing by peak0_c calibrates Q_c*P_c^2 to the SAME order as the self term
    # eta_c*P_c^3 when every channel sits near its own peak (zeta ~ O(1) tunes how much
    # a neighbour's power matters relative to your own) -- on SHORT (low-power)
    # instances the whole nonlinear term is small so this barely bites, but on TRAP
    # instances (every neighbour also near/above its own peak) it compounds the
    # self-nonlinearity collapse -- planted structure a per-channel-only view misses.
    zeta = [rnd.uniform(0.15, 0.6) for _ in range(C)]
    kappa = [[0.0] * C for _ in range(C)]
    for c in range(C):
        for c2 in range(C):
            if c == c2:
                continue
            kappa[c][c2] = zeta[c] * eta[c] / max(peak0_c[c], 1.0) / (1.0 + abs(c - c2))

    baud = rnd.randint(20, 40)

    out = []
    out.append(f"{C} {S}")
    out.append(" ".join(str(a) for a in ase))
    out.append(" ".join("%.9f" % e for e in eta))
    out.append(str(K))
    for k in range(K):
        out.append(f"{BPS[k]} {REQ_SNR[k]:.6f}")
    out.append(" ".join("%.6f" % p for p in pmax))
    out.append(str(baud))
    for c in range(C):
        out.append(" ".join("%.9f" % kappa[c][c2] for c2 in range(C)))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    gen(int(sys.argv[1]))


if __name__ == "__main__":
    main()
