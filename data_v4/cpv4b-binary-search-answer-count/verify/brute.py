import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    p = []
    for _ in range(m):
        p.append(int(data[idx])); idx += 1

    # Obviously-correct brute: walk minutes 1,2,3,... and count those
    # divisible by at least one p_i, stopping at the k-th such minute.
    # (Only used on SMALL cases where the answer is tiny.)
    count = 0
    x = 0
    while True:
        x += 1
        lit = any(x % pi == 0 for pi in p)
        if lit:
            count += 1
            if count == k:
                print(x)
                return

if __name__ == "__main__":
    main()
