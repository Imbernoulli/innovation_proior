import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    D = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    # Exhaustive: enumerate every contiguous block [l, r] of length >= 2,
    # compute max - min directly, count those with max - min <= D.
    count = 0
    for l in range(n):
        cur_max = a[l]
        cur_min = a[l]
        for r in range(l + 1, n):
            if a[r] > cur_max:
                cur_max = a[r]
            if a[r] < cur_min:
                cur_min = a[r]
            if cur_max - cur_min <= D:
                count += 1
    print(count)

main()
