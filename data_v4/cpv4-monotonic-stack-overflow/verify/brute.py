import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + k]) for k in range(n)]
    idx += n

    # Sum over ALL subarrays of the minimum element. Obviously correct O(n^2).
    total = 0
    for i in range(n):
        cur_min = None
        for j in range(i, n):
            if cur_min is None or a[j] < cur_min:
                cur_min = a[j]
            total += cur_min
    print(total)

if __name__ == "__main__":
    main()
