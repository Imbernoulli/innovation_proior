# TIER: strong
# Search for a LARGE progression-free difference set A in Z_n (deterministic,
# seed-swept insertion orders), then build the product corner-free set
# T = {(x,y): (x-y) mod n in A} minus blocked cells.  A larger A packs many more
# relays than the few diagonals of the greedy tier while staying corner-free.
import sys, random


def ap_free_add(Aset, a, n):
    # can 'a' be added to progression-free set Aset without creating a 3-AP?
    for p in Aset:
        # a as the top element: p, ?, a  -> middle m with 2m = p+a
        # and generally check any AP created by adding a
        pass
    A2 = Aset | {a}
    for p in A2:
        for q in A2:
            if p == q:
                continue
            r = (2 * q - p) % n
            if r in A2 and r != p and r != q:
                return False
    return True


def build_A(n, order):
    A = set()
    for a in order:
        if ap_free_add(A, a, n):
            A.add(a)
    return A


def main():
    t = sys.stdin.read().split()
    i = 0
    n = int(t[i]); i += 1
    k = int(t[i]); i += 1
    blocked = set()
    for _ in range(k):
        blocked.add((int(t[i]), int(t[i + 1]))); i += 2

    best = build_A(n, list(range(n)))
    for seed in range(40):
        order = list(range(n))
        random.Random(seed * 131 + 1).shuffle(order)
        A = build_A(n, order)
        if len(A) > len(best):
            best = A
    A = best

    S = []
    for x in range(n):
        for y in range(n):
            if (x - y) % n in A and (x, y) not in blocked:
                S.append((x, y))

    out = [str(len(S))]
    for (x, y) in S:
        out.append("%d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
