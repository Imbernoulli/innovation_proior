import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = []
    for i in range(n):
        a.append(int(data[idx])); idx += 1
    # Independent brute force: explicit double loop over all unordered pairs.
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += a[i] * a[j]
    print(total)

main()
