# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    balance = [0] * (n + 1)
    for _ in range(m):
        d = int(next(it)); c = int(next(it)); a = int(next(it))
        balance[d] -= a
        balance[c] += a

    # Naive baseline: a single ascending-party-id prefix-carry chain. No attempt
    # to discover any zero-sum grouping -- just walk the ids in order and pass
    # the running carry to the next party.
    transfers = []
    carry = 0
    for i in range(1, n):
        carry += balance[i]
        if carry != 0:
            # transfer (u, v, amt): u pays v amt -> contributes -amt to u, +amt to v.
            # prefix>0 -> party i+1 pays party i; prefix<0 -> party i pays party i+1.
            if carry > 0:
                transfers.append((i + 1, i, carry))
            else:
                transfers.append((i, i + 1, -carry))

    out = [str(len(transfers))]
    for u, v, a in transfers:
        out.append(f"{u} {v} {a}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
