# TIER: greedy
# The "obvious" actuarial recipe: fit the expected loss of each tier's CURRENT
# (pre-selection) blended pool, add a flat underwriting margin, clip to the
# rate-change-cap band. This is cost-plus / actuarially-fair pricing on the pool
# you have TODAY -- it never asks who is still there tomorrow.
import sys

MARGIN = 1.15


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it))
    prices = []
    for _ in range(K):
        p0 = int(next(it)); c = int(next(it)); cap = int(next(it)); B = int(next(it))
        total_n = 0
        total_loss = 0.0
        for _ in range(B):
            n = int(next(it)); m = int(next(it))
            vs = []; prs = []
            for _ in range(m):
                vs.append(int(next(it))); prs.append(int(next(it)))
            next(it); next(it)  # tlo thi -- greedy ignores elasticity entirely
            eloss = sum(v * p for v, p in zip(vs, prs)) / 1000.0
            total_n += n
            total_loss += n * eloss
        blended = total_loss / max(1, total_n)
        target = round(blended * MARGIN)
        delta = (p0 * cap) // 100
        lo, hi = max(0, p0 - delta), p0 + delta
        price = min(hi, max(lo, target))
        prices.append(price)
    sys.stdout.write("\n".join(str(p) for p in prices) + "\n")


if __name__ == "__main__":
    main()
