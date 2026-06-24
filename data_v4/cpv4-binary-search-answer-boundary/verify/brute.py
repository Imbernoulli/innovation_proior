import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    b = []
    for _ in range(n):
        b.append(int(data[idx])); idx += 1

    # Impossibility: at most floor(n/k) bouquets ever.
    if m * k > n:
        print(-1)
        return

    def bouquets_on(t):
        # count bouquets if today is day t: a bed is bloomed iff b[i] <= t
        total = 0
        run = 0
        for i in range(n):
            if b[i] <= t:
                run += 1
                if run == k:
                    total += 1
                    run = 0
            else:
                run = 0
        return total

    # Brute force: try every distinct candidate day in sorted order; the earliest
    # feasible day must be one of the bloom days. Sort the distinct bloom days and
    # return the first that yields >= m bouquets.
    days = sorted(set(b))
    for t in days:
        if bouquets_on(t) >= m:
            print(t)
            return
    # Guaranteed reachable since m*k <= n; max day yields one full run.
    print(days[-1])

main()
