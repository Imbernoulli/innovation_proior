# TIER: greedy
# Classic ABC / velocity slotting: count each SKU's GLOBAL pick frequency pooled
# over every order in every scenario (ignoring which scenario an order belongs
# to, and ignoring which OTHER SKUs it is co-purchased with), then place the
# highest-frequency SKUs in the slots closest to the depot (lowest aisle index,
# then lowest depth). This is the textbook first move and it treats every SKU
# independently -- it never notices that a rarely-purchased SKU can dominate the
# worst-case (p90) scenario cost because of who it is bought ALONGSIDE.
import sys

def main():
    d = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = d[p]; p += 1
        return v
    N = int(nxt()); A = int(nxt()); L = int(nxt()); K = int(nxt()); W = int(nxt())

    freq = [0] * N
    for _ in range(K):
        q = int(nxt())
        for _ in range(q):
            s = int(nxt())
            for _ in range(s):
                sku = int(nxt())
                freq[sku] += 1

    order = sorted(range(N), key=lambda sku: (-freq[sku], sku))
    perm = [0] * N
    for slot, sku in enumerate(order):
        perm[sku] = slot

    print(N)
    print("\n".join(str(x) for x in perm))

main()
