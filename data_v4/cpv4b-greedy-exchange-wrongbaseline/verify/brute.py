import sys
from itertools import combinations

def feasible(jobs):
    # jobs: list of (t, d). Try EDD order; check finishes <= deadline.
    # (EDD optimality is what we are testing the greedy against, so brute
    #  does NOT rely on it: it tries ALL orderings of the subset.)
    from itertools import permutations
    for perm in permutations(jobs):
        clock = 0
        ok = True
        for (t, d) in perm:
            clock += t
            if clock > d:
                ok = False
                break
        if ok:
            return True
    return False

def solve(jobs):
    n = len(jobs)
    best = 0
    # try every subset, largest first would be faster but n is tiny
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            sub = [jobs[i] for i in combo]
            if feasible(sub):
                best = max(best, k)
    return best

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    jobs = []
    for _ in range(n):
        t = int(data[idx]); d = int(data[idx + 1]); idx += 2
        jobs.append((t, d))
    print(solve(jobs))

if __name__ == "__main__":
    main()
