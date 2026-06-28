import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); q = int(next(it))
    px = [0]*n; py = [0]*n; w = [0]*n
    for i in range(n):
        px[i] = int(next(it)); py[i] = int(next(it)); w[i] = int(next(it))

    out = []
    last = 0
    for _ in range(q):
        t = int(next(it))
        if t == 1:
            idx = int(next(it)); d = int(next(it))
            w[idx] += d
        else:
            a = int(next(it)); b = int(next(it)); c = int(next(it)); e = int(next(it))
            X1 = a ^ last; Y1 = b ^ last; X2 = c ^ last; Y2 = e ^ last
            s = 0
            if X1 <= X2 and Y1 <= Y2:
                for i in range(n):
                    if X1 <= px[i] <= X2 and Y1 <= py[i] <= Y2:
                        s += w[i]
            last = s
            out.append(str(s))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
