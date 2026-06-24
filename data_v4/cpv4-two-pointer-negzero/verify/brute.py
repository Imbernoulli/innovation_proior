import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    if idx >= len(data):
        # no threshold provided; treat as 0 pairs (defensive, generator always supplies T)
        print(0)
        return
    T = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    # Obvious O(n^2): enumerate every unordered pair and test the threshold directly.
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] + a[j] >= T:
                count += 1
    print(count)

main()
