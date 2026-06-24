import sys

def main():
    data = sys.stdin.buffer.read().split()
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
            l = int(data[idx]); idx += 1
            r = int(data[idx]); idx += 1
            v = int(data[idx]); idx += 1
            # 1-indexed inclusive; add v to each element
            for i in range(l - 1, r):
                a[i] += v
        else:  # t == 2, range sum
            l = int(data[idx]); idx += 1
            r = int(data[idx]); idx += 1
            s = 0
            for i in range(l - 1, r):
                s += a[i]
            out.append(str(s))
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")

main()
