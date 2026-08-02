# TIER: trivial
# Reproduces the checker's own baseline: find the single rail with the lowest
# AVERAGE fee across ticket buckets, and route every issuer segment / bucket
# to it with a single attempt (no cascade at all -- if declined, the payment
# is simply lost). Ignores auth rates and retries entirely.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    R = int(it[p]); S = int(it[p + 1]); B = int(it[p + 2]); K = int(it[p + 3]); FAILPEN_BPS = int(it[p + 4]); p += 5
    amt = [int(it[p + i]) for i in range(B)]; p += B
    fixedFee = [[0] * B for _ in range(R)]
    pctBps = [[0] * B for _ in range(R)]
    for r in range(R):
        for b in range(B):
            fixedFee[r][b] = int(it[p]); pctBps[r][b] = int(it[p + 1]); p += 2
        p += 1  # retry surcharge, unused
    p += S * R   # auth rates, unused
    p += S * B   # volumes, unused

    def fee(r, b):
        return fixedFee[r][b] + pctBps[r][b] * amt[b] / 10000.0

    avg_fee = [sum(fee(r, b) for b in range(B)) / B for r in range(R)]
    best_r = min(range(R), key=lambda r: (avg_fee[r], r))

    out = []
    for _s in range(S):
        for _b in range(B):
            out.append("1 %d" % best_r)
    print("\n".join(out))


if __name__ == "__main__":
    main()
