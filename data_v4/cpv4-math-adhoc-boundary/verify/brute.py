import sys

def solve(n, a, b):
    if n <= 0:
        return 0
    if a > b:
        return 0
    cnt = 0
    for x in range(1, n + 1):
        q = n // x  # floor(n/x), n,x > 0
        if a <= q <= b:
            cnt += 1
    return cnt

def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        n = int(data[idx]); a = int(data[idx+1]); b = int(data[idx+2]); idx += 3
        out.append(str(solve(n, a, b)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
