# TIER: strong
"""
The insight: load balance is a property of BATCHES, not totals. Instead of
placing each dish's R replicas by a frequency-balanced rule that is blind to
which dishes are actually plated together, read the whole course trace first
and place replicas to maximize per-batch routing FLEXIBILITY -- an
antiaffinity design: a dish's replicas should avoid kitchens already claimed
by the OTHER dishes it is co-plated with, so that whichever course arrives,
its dishes have many distinct kitchens to be routed across.

Algorithm (co-occurrence-aware antiaffinity placement):
  1. From the trace, weight each dish by its total "crowding exposure" --
     the sum, over every course containing it, of (course size - 1).
  2. Place dishes in decreasing exposure order (the most contended dishes get
     first pick of open kitchens).
  3. For dish d, maintain -- incrementally, for every course c that contains
     d -- a per-course counter of which kitchens are already used by OTHER,
     already-placed dishes of that course. Sum these counters across all of
     d's courses to get a per-kitchen "conflict score". Choose the R kitchens
     with (conflict score, current global load, index) smallest, subject to
     remaining capacity.
  4. Update the running counters and capacities, continue.

Routing is then the SAME per-course greedy used by the "greedy" reference
(serve from whichever of a dish's own replicas is least loaded so far in that
course) -- the entire win comes from the placement, isolating the hook.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = data[idx]
        idx += 1
        return v

    K = int(nxt()); R = int(nxt()); D = int(nxt()); B = int(nxt())
    cap = [int(nxt()) for _ in range(K)]
    courses = []
    for _ in range(B):
        s = int(nxt())
        dishes = [int(nxt()) for _ in range(s)]
        courses.append(dishes)

    courses_of = [[] for _ in range(D)]
    for c, dishes in enumerate(courses):
        for d in dishes:
            courses_of[d].append(c)

    weight = [0] * D
    for d in range(D):
        w = 0
        for c in courses_of[d]:
            w += len(courses[c]) - 1
        weight[d] = w

    order = sorted(range(D), key=lambda d: (-weight[d], d))

    remaining_cap = cap[:]
    global_load = [0] * K
    course_usage = [dict() for _ in range(B)]  # course_usage[c][k] -> #placed dishes of c using k
    replica = [None] * D

    for d in order:
        conflict = [0] * K
        for c in courses_of[d]:
            cu = course_usage[c]
            if cu:
                for k, cnt in cu.items():
                    conflict[k] += cnt

        open_ks = [k for k in range(K) if remaining_cap[k] > 0]
        if len(open_ks) < R:
            open_ks = list(range(K))  # safety fallback, should not trigger
        open_ks.sort(key=lambda k: (conflict[k], global_load[k], k))
        chosen = open_ks[:R]

        replica[d] = chosen
        for k in chosen:
            remaining_cap[k] -= 1
            global_load[k] += 1
        for c in courses_of[d]:
            cu = course_usage[c]
            for k in chosen:
                cu[k] = cu.get(k, 0) + 1

    out = []
    for d in range(D):
        out.append(" ".join(map(str, replica[d])))

    for dishes in courses:
        course_load = [0] * K
        line = []
        for d in dishes:
            best_k, best_load = None, None
            for k in replica[d]:
                cl = course_load[k]
                if best_load is None or cl < best_load or (cl == best_load and k < best_k):
                    best_load, best_k = cl, k
            course_load[best_k] += 1
            line.append(best_k)
        out.append(" ".join(map(str, line)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
