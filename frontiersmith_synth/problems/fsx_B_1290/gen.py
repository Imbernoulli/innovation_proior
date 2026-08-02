import sys, random

# ---------------------------------------------------------------------------
# payment-routing-cost (format C, MINIMIZE cost per successful payment)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.
#   Deterministic in testId only (seed = f(testId)).
#
# Instance:
#   line 1:            R S B K FAILPEN_BPS
#   line 2:            amt[0] .. amt[B-1]                (representative ticket size per bucket, cents)
#
# failpen(b) = FAILPEN_BPS * amt[b] // 10000  (a lost payment costs a fixed
# FRACTION of the ticket value -- proportional lost-margin/churn model, so it
# scales sanely across ticket-size buckets instead of one flat constant that
# would dwarf small tickets and be trivial next to large ones).
#   next R lines:      fixedFee[r][0] pctBps[r][0] ... fixedFee[r][B-1] pctBps[r][B-1] retrySurcharge[r]
#   next S lines:      auth[s][0] .. auth[s][R-1]        (issuer-segment auth rate per rail, 4dp)
#   next S lines:      vol[s][0]  .. vol[s][B-1]          (transaction volume per bucket, ints)
#
# Mechanisms composed:
#   - interchange-fee-tiers : fee(rail,bucket) = fixedFee + pctBps*amt[bucket]/10000, tiered by ticket size
#   - auth-rate-by-issuer   : authorization probability depends on (rail, issuer segment) jointly
#   - retry-cascade-cost    : every attempt after the first on a cell pays an extra rail-specific
#                             retry surcharge (re-authorization / gateway overhead)
#
# Trap (planted on TRAP_IDS, >=3 of 10 cases): rail 0 is a "budget" rail -- cheapest fee AND
# cheapest retry surcharge -- but its authorization rate is only good for ONE friendly issuer
# segment and poor (0.20-0.35) for every other segment. A fee-only cascade always tries rail 0
# first for every segment, wasting an attempt (and its retry surcharge on the next hop) on
# segments where it is very likely to decline, before ever reaching a rail that actually works.
# ---------------------------------------------------------------------------

R_LADDER = [3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
S_LADDER = [2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
B = 3
AMT = [2000, 8000, 50000]          # representative ticket sizes per bucket, in cents
TRAP_IDS = {4, 5, 6, 7, 8, 9, 10}   # 7 of 10 cases plant the auth-rate trap; 1-3 are warm-ups
FAILPEN_BPS = 220                   # lost-payment penalty = 2.2% of ticket value


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    rng = random.Random(50021 + 977 * t)

    R = R_LADDER[t - 1]
    S = S_LADDER[t - 1]
    K = min(R, 4)
    trap = t in TRAP_IDS

    # ---- fee schedule: fixedFee + pctBps, growing with rail index (pricier rails) ----
    fixedFee = [[0] * B for _ in range(R)]
    pctBps = [[0] * B for _ in range(R)]
    retry = [0] * R
    for r in range(R):
        for b in range(B):
            fixedFee[r][b] = 6 + 5 * r + 2 * b
            pctBps[r][b] = max(20, 90 + 35 * r - 4 * b)
        retry[r] = 8 + 4 * r

    # ---- auth rates: rail 0 is cheap-but-narrow under the trap ----
    auth = [[0.0] * R for _ in range(S)]
    for s in range(S):
        for r in range(R):
            if trap and r == 0:
                base = 0.90 if s == 0 else 0.25
                jit = rng.uniform(-0.05, 0.05)
                lo, hi = (0.80, 0.98) if s == 0 else (0.10, 0.42)
            else:
                base = 0.86 + 0.025 * (r % 4)
                jit = rng.uniform(-0.04, 0.04)
                lo, hi = 0.55, 0.98
            auth[s][r] = round(clamp(base + jit, lo, hi), 4)

    # ---- transaction volume mix per (segment, bucket) ----
    vol = [[rng.randint(50, 400) for _ in range(B)] for _ in range(S)]

    out = []
    out.append("%d %d %d %d %d" % (R, S, B, K, FAILPEN_BPS))
    out.append(" ".join(str(a) for a in AMT))
    for r in range(R):
        row = []
        for b in range(B):
            row.append(str(fixedFee[r][b]))
            row.append(str(pctBps[r][b]))
        row.append(str(retry[r]))
        out.append(" ".join(row))
    for s in range(S):
        out.append(" ".join("%.4f" % auth[s][r] for r in range(R)))
    for s in range(S):
        out.append(" ".join(str(v) for v in vol[s]))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
