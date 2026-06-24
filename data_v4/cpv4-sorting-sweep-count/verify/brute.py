import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    D = int(data[idx]); idx += 1
    p = []
    for _ in range(n):
        p.append(int(data[idx])); idx += 1
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(p[i] - p[j])
            circ = min(d, L - d)
            if circ <= D:
                cnt += 1
    print(cnt)

main()
