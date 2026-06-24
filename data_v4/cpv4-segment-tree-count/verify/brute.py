import sys

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    h = []
    for _ in range(n):
        h.append(int(data[idx])); idx += 1
    out = []
    for _ in range(q):
        t = int(data[idx]); idx += 1
        if t == 1:
            p = int(data[idx]); idx += 1
            x = int(data[idx]); idx += 1
            h[p] = x
        else:
            l = int(data[idx]); idx += 1
            r = int(data[idx]); idx += 1
            # Count strict prefix maxima of h[l..r]: positions strictly greater
            # than every element to their left within the window.
            cnt = 0
            cur = None
            for k in range(l, r + 1):
                if cur is None or h[k] > cur:
                    cnt += 1
                    cur = h[k]
            out.append(str(cnt))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
