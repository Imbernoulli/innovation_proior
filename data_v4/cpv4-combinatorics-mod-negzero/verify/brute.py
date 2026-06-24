import sys

# Independent brute force: enumerate every non-empty subset, compute the actual
# sign of its product, count those that are strictly positive, reduce mod m.
def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    count = 0
    for mask in range(1, 1 << n):          # non-empty subsets only
        sign = 1
        zero = False
        for i in range(n):
            if mask & (1 << i):
                v = a[i]
                if v == 0:
                    zero = True
                    break
                if v < 0:
                    sign = -sign
        if not zero and sign > 0:
            count += 1

    print(count % m)

if __name__ == "__main__":
    main()
