# TIER: strong
# Insight: growth is killed EXACTLY by proper-subgroup containment (Dickson's
# classification of SL2(F_p) subgroups), so instead of trusting any single fixed
# recipe (the registry hint, or a textbook generator pair), we CERTIFY escape from
# every small subgroup empirically: build a small deterministic portfolio of short
# words in two fixed SL2 seeds (products of det-1 matrices stay det-1, no modular
# inverse needed) plus the registry hint itself, then directly MEASURE each
# candidate k-subset's own Cayley-ball growth to radius r (mirroring exactly what
# the checker will do) and keep the subset with the largest verified flood. This
# is the real optimization: certifying genuine (non-trapped) generation, after
# which growth follows almost for free.
import sys
import itertools


def mat_inv(M, p):
    a, b, c, d = M
    return (d % p, (-b) % p, (-c) % p, a % p)


def mat_mul(A, B, p):
    a, b, c, d = A
    e, f, g, h = B
    return ((a * e + b * g) % p, (a * f + b * h) % p,
            (c * e + d * g) % p, (c * f + d * h) % p)


def ball_size(gens, r, p):
    S = set()
    for M in gens:
        S.add(M)
        S.add(mat_inv(M, p))
    ident = (1, 0, 0, 1)
    visited = {ident}
    frontier = {ident}
    for _ in range(r):
        nxt = set()
        for M in frontier:
            for s in S:
                nm = mat_mul(M, s, p)
                if nm not in visited:
                    visited.add(nm)
                    nxt.add(nm)
        frontier = nxt
    return len(visited)


def build_pool(p, hint):
    P = (1, 1, 1, 2)
    Q = (2, 1, 1, 1)
    pool = [P, Q,
            mat_mul(P, P, p), mat_mul(Q, Q, p),
            mat_mul(P, Q, p), mat_mul(Q, P, p),
            mat_mul(mat_mul(P, Q, p), P, p),
            mat_mul(mat_mul(Q, P, p), Q, p)]
    pool += hint
    return pool


def main():
    data = [ln for ln in sys.stdin.read().split("\n") if ln.strip() != ""]
    p, k, r = map(int, data[0].split())
    hint = []
    for ln in data[1:1 + k]:
        a, b, c, d = map(int, ln.split())
        hint.append((a % p, b % p, c % p, d % p))

    pool = build_pool(p, hint)
    # dedupe while keeping order (small pool, cheap)
    seen = set()
    uniq_pool = []
    for m in pool:
        if m not in seen:
            seen.add(m)
            uniq_pool.append(m)

    best_gens = uniq_pool[:k]
    best_F = -1
    for combo in itertools.combinations(range(len(uniq_pool)), k):
        gens = [uniq_pool[i] for i in combo]
        F = ball_size(gens, r, p)
        if F > best_F:
            best_F = F
            best_gens = gens

    print("\n".join(f"{a} {b} {c} {d}" for (a, b, c, d) in best_gens))


if __name__ == "__main__":
    main()
