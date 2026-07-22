# TIER: greedy
"""
The "obvious" strong-coder solution: place replicas with a frequency-balanced,
consistent-hashing-style scheme (dish d's R replicas = kitchens
[(d%G)*R, (d%G)*R+R), G=K//R) -- this balances TOTAL replica counts across
kitchens PERFECTLY. Then, since a fixed replica set alone can't help a busy
course, do the sensible per-course thing: for each dish in a course (in the
order given), route it to whichever of its OWN replica kitchens currently has
the smallest load *within that course*.

This is a reasonable, natural pipeline -- but the placement never looks at
which dishes are actually co-plated, so on courses drawn from a single
frequency-balance residue class, every dish's replica set is IDENTICAL and no
amount of clever routing has anywhere to spread to.
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

    G = max(1, K // R)
    replica = []
    out = []
    for d in range(D):
        start = (d % G) * R
        rs = [start + i for i in range(R)]
        replica.append(rs)
        out.append(" ".join(map(str, rs)))

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
