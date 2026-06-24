import sys

MOD = 1000000007

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    full = (1 << n) - 1 if n > 0 else 0
    allowed = set()
    for _ in range(m):
        x = int(data[idx]); idx += 1
        x &= full
        if x == 0:
            continue
        allowed.add(x)

    # Fully independent brute force: enumerate EVERY set partition of
    # {0,1,...,n-1} via restricted growth strings (RGS). An RGS a[0..n-1] with
    # a[0]=0 and a[i] <= 1 + max(a[0..i-1]) is in bijection with the unordered
    # set partitions (each partition appears exactly once). For each partition
    # we compute the bitmask of each block and count the partition iff every
    # block's mask is an allowed candidate squad.
    if n == 0:
        # The only partition of the empty set is the empty partition (1 way),
        # using no squads. So the answer is 1.
        print(1)
        return

    total = 0
    a = [0] * n  # restricted growth string

    # iterate over all RGS by incrementing like a mixed-radix counter
    def gen(pos, mx):
        nonlocal total
        if pos == n:
            # build block masks
            blocks = {}
            for i in range(n):
                blocks[a[i]] = blocks.get(a[i], 0) | (1 << i)
            ok = True
            for mask in blocks.values():
                if mask not in allowed:
                    ok = False
                    break
            if ok:
                total += 1
            return
        for v in range(0, mx + 2):
            a[pos] = v
            gen(pos + 1, max(mx, v))

    a[0] = 0
    gen(1, 0)
    print(total % MOD)

if __name__ == "__main__":
    main()
