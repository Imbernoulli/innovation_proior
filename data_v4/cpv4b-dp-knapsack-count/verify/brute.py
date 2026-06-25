import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    S = int(data[idx]); idx += 1
    MOD = int(data[idx]); idx += 1
    v = []
    c = []
    for i in range(n):
        v.append(int(data[idx])); idx += 1
        c.append(int(data[idx])); idx += 1

    # Brute force: enumerate the number of copies k_i in [0, c_i] for each
    # denomination i. Each distinct tuple (k_0, k_1, ..., k_{n-1}) is exactly
    # one multiset (combination) of stamps. Count those whose total == S.
    # This directly counts unordered selections, no ordering involved.
    count = 0

    def rec(i, remaining):
        nonlocal count
        if i == n:
            if remaining == 0:
                count += 1
            return
        val = v[i]
        if val <= 0:
            # zero-value stamps do not change the sum; per the contract they
            # are inert. Treat exactly like the solution: they cannot be used
            # to alter the sum, so contribute a single "zero copies" branch.
            rec(i + 1, remaining)
            return
        k = 0
        while k <= c[i] and k * val <= remaining:
            rec(i + 1, remaining - k * val)
            k += 1

    rec(0, S)
    print(count % MOD)

main()
