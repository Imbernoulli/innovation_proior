import sys

MOD = 998244353

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    V = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Frequency polynomial coefficients.
    f = [0] * (V + 1)
    for x in a:
        f[x] = (f[x] + 1) % MOD

    # Naive O((V+1)^2) convolution: f2 = f * f (number of ordered pairs summing to s).
    f2 = [0] * (2 * V + 1)
    for i in range(V + 1):
        if f[i] == 0:
            continue
        fi = f[i]
        for j in range(V + 1):
            if f[j]:
                f2[i + j] = (f2[i + j] + fi * f[j]) % MOD

    # Naive convolution f3 = f2 * f (number of ordered triples summing to s).
    f3 = [0] * (3 * V + 1)
    for i in range(2 * V + 1):
        if f2[i] == 0:
            continue
        c = f2[i]
        for j in range(V + 1):
            if f[j]:
                f3[i + j] = (f3[i + j] + c * f[j]) % MOD

    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        s = int(data[idx]); idx += 1
        ans = f3[s] if 0 <= s < len(f3) else 0
        out.append(str(ans % MOD))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
