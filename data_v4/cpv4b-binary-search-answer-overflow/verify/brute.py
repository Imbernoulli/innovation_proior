import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    P = int(data[idx]); idx += 1
    s = []
    for _ in range(n):
        s.append(int(data[idx])); idx += 1

    # Brute force: simulate time T = 0, 1, 2, ... and count total pages printed
    # by time T = sum_i floor(T / s[i]). Return the first T whose count >= P.
    # Independent of the binary-search logic: a plain linear scan.
    T = 0
    while True:
        total = 0
        for period in s:
            total += T // period
        if total >= P:
            print(T)
            return
        T += 1

main()
