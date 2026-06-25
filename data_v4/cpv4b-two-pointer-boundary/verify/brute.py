import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    D = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Brute force: enumerate every contiguous subarray [i, j], compute max - min,
    # count it if strictly less than D.
    count = 0
    for i in range(n):
        cur_max = a[i]
        cur_min = a[i]
        for j in range(i, n):
            if a[j] > cur_max:
                cur_max = a[j]
            if a[j] < cur_min:
                cur_min = a[j]
            if cur_max - cur_min < D:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
