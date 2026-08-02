# TIER: greedy
# The obvious "recipe": route to the lowest-FEE rail first, then the next
# cheapest, and so on, up to the attempt cap -- optimizing cost PER ATTEMPT.
# Same fee-sorted cascade for every issuer segment; auth-rate-by-issuer is
# never consulted, so a segment where the cheap rail declines often just
# burns extra retry-cascade cost before the cascade happens to reach a rail
# that actually authorizes.
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
        p += 1  # retry surcharge, unused by this tier
    p += S * R   # auth rates, unused by this tier
    p += S * B   # volumes, unused by this tier

    def fee(r, b):
        return fixedFee[r][b] + pctBps[r][b] * amt[b] / 10000.0

    out = []
    for _s in range(S):
        for b in range(B):
            order = sorted(range(R), key=lambda r: (fee(r, b), r))[:K]
            out.append(str(len(order)) + " " + " ".join(str(r) for r in order))
    print("\n".join(out))


if __name__ == "__main__":
    main()
