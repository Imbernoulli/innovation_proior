import sys

# Independent brute force for the gig-scheduling problem.
# Method: enumerate EVERY subset of gigs. A set of gigs (each needing one
# distinct day in 1..deadline, one gig per day) is feasible iff, sorting the
# chosen deadlines ascending d_1 <= d_2 <= ... <= d_m, we have d_j >= j for all
# j (the j-th earliest-deadline gig can sit on day j). This is the standard
# Hall-condition feasibility test for deadline scheduling -- obviously correct,
# and a completely different method from the DSU-greedy in sol.cpp.

def solve(d, v):
    n = len(d)
    best = 0  # empty subset is always allowed
    for mask in range(1 << n):
        deadlines = []
        total = 0
        for i in range(n):
            if mask & (1 << i):
                deadlines.append(d[i])
                total += v[i]
        deadlines.sort()
        feasible = True
        for j, dl in enumerate(deadlines, start=1):
            if dl < j:          # the j-th gig has no free day
                feasible = False
                break
        if feasible and total > best:
            best = total
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    d = [int(data[idx + i]) for i in range(n)]; idx += n
    v = [int(data[idx + i]) for i in range(n)]; idx += n
    print(solve(d, v))


if __name__ == "__main__":
    main()
