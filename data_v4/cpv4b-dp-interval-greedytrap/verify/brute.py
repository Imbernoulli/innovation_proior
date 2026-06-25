import sys

def overlap(a, b):
    # half-open intervals [s, e): they conflict iff they share an interior point,
    # i.e. they overlap iff a.s < b.e and b.s < a.e.
    return a[0] < b[1] and b[0] < a[1]

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    iv = []
    for _ in range(n):
        s = int(data[idx]); e = int(data[idx+1]); v = int(data[idx+2]); idx += 3
        iv.append((s, e, v))

    best = 0
    for mask in range(1 << n):
        chosen = [iv[i] for i in range(n) if (mask >> i) & 1]
        ok = True
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if overlap(chosen[i], chosen[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            total = sum(c[2] for c in chosen)
            if total > best:
                best = total
    print(best)

main()
