# TIER: trivial
"""
Reproduces the checker's own internal baseline EXACTLY: a frequency-balanced
block placement (dish d's R replicas = kitchens [(d%G)*R, (d%G)*R+R)) with
G = K // R), and naive routing that always serves from replica #0 -- no
per-course adaptivity, no antiaffinity awareness at all.
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
        line = [str(replica[d][0]) for d in dishes]
        out.append(" ".join(line))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
