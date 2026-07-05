import sys, random

def main():
    i = int(sys.argv[1])
    rng = random.Random(20265 + 101 * i)

    # ---- difficulty / structure ladder (small scale) ----
    # side grows 3 -> 12; the cut budget k and the latency band both widen so that
    # forced detours become progressively more expensive on harder tests.
    side = 3 + (i - 1)             # 3,4,...,12
    R = side
    C = side
    n = R * C
    k = max(3, side)               # removable-link budget
    w_lo = 1
    w_hi = 10 + 2 * i              # 12 .. 30

    def nid(r, c):
        return r * C + c

    edges = []  # (u, v, w)
    # horizontal links
    for r in range(R):
        for c in range(C - 1):
            edges.append((nid(r, c), nid(r, c + 1), rng.randint(w_lo, w_hi)))
    # vertical links
    for r in range(R - 1):
        for c in range(C):
            edges.append((nid(r, c), nid(r + 1, c), rng.randint(w_lo, w_hi)))

    m = len(edges)
    s = nid(0, 0)              # top-left relay
    t = nid(R - 1, C - 1)      # bottom-right relay (opposite corner)

    out = [f"{n} {m} {k} {s} {t}"]
    for (u, v, w) in edges:
        out.append(f"{u} {v} {w}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
