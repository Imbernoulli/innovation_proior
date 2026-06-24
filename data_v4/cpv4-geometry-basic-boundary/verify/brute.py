import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    covered = set()
    for _ in range(n):
        a = int(data[idx]); b = int(data[idx+1]); c = int(data[idx+2]); d = int(data[idx+3])
        idx += 4
        x1, x2 = min(a, c), max(a, c)
        y1, y2 = min(b, d), max(b, d)
        # Mark every integer lattice point inside or on the boundary (closed rectangle).
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                covered.add((x, y))
    print(len(covered))

if __name__ == "__main__":
    main()
