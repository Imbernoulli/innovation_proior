import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    xs = []
    ys = []
    for _ in range(n):
        x = int(data[idx]); idx += 1
        y = int(data[idx]); idx += 1
        xs.append(x); ys.append(y)
    best = 0
    # O(n^2) all-pairs squared distance; obviously correct.
    for i in range(n):
        xi = xs[i]; yi = ys[i]
        for j in range(i + 1, n):
            dx = xi - xs[j]
            dy = yi - ys[j]
            d2 = dx * dx + dy * dy
            if d2 > best:
                best = d2
    print(best)

main()
