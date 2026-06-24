import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    out = []
    for _ in range(q):
        t = int(data[idx]); idx += 1
        if t == 1:
            p = int(data[idx]); idx += 1
            v = int(data[idx]); idx += 1
            a[p - 1] = v
        else:
            l = int(data[idx]); idx += 1
            r = int(data[idx]); idx += 1
            # maximum-sum contiguous block within a[l-1 .. r-1], empty block (sum 0) allowed
            best = 0  # empty selection
            # O((r-l+1)^2) brute force over all subarrays
            for i in range(l - 1, r):
                s = 0
                for j in range(i, r):
                    s += a[j]
                    if s > best:
                        best = s
            out.append(str(best))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
