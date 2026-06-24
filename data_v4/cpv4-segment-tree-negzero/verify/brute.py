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
            i = int(data[idx]); idx += 1
            x = int(data[idx]); idx += 1
            a[i] = x
        else:
            l = int(data[idx]); idx += 1
            r = int(data[idx]); idx += 1
            # maximum sum of a NON-EMPTY contiguous subarray within a[l..r]
            best = None
            for s in range(l, r + 1):
                cur = 0
                for e in range(s, r + 1):
                    cur += a[e]
                    if best is None or cur > best:
                        best = cur
            out.append(str(best))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
