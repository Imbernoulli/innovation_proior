# TIER: greedy
# "Safe textbook shim": for each OLD type, try every SINGLE new type as a direct 1:1
# substitute; if one reproduces the exact transform, use it. If none does (the old type
# has "no counterpart"), REFUSE to emulate it (emit nothing, L=0) rather than risk sending
# something wrong. This is the obvious, defensible engineering choice -- and it is a trap:
# refusing on the unmappable types corrupts the shim's own carried-forward observable
# state, which then poisons every later checkpoint in that session too.
import sys


def matvec_step(A, b, v, mod):
    n = len(v)
    r = len(A)
    out = [0] * r
    for i in range(r):
        Ai = A[i]
        s = 0
        for j in range(n):
            s += Ai[j] * v[j]
        out[i] = (s + b[i]) % mod
    return out


def realize1(A_new, b_new, s, K_OLD, K_EXTRA, P):
    v0 = [0] * K_OLD + [0] * K_EXTRA  # symbolic: compute matrix action via basis probing
    # Build the K_OLD x K_OLD matrix / vector directly from A_new[s], b_new[s]:
    A1 = [row[:K_OLD] for row in A_new[s][:K_OLD]]
    b1 = b_new[s][:K_OLD]
    return A1, b1


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    K_OLD = int(nxt()); K_NEW = int(nxt()); M_OLD = int(nxt()); M_NEW = int(nxt())
    P = int(nxt()); L_MAX = int(nxt()); _cn = int(nxt()); _cd = int(nxt())
    K_EXTRA = K_NEW - K_OLD
    assert nxt() == "NEWTYPES"
    A_new, b_new = [], []
    for _ in range(M_NEW):
        A = [[int(nxt()) for _ in range(K_NEW)] for _ in range(K_NEW)]
        b = [int(nxt()) for _ in range(K_NEW)]
        A_new.append(A); b_new.append(b)
    assert nxt() == "OLDTYPES"
    A_old, b_old = [], []
    for _ in range(M_OLD):
        A = [[int(nxt()) for _ in range(K_OLD)] for _ in range(K_OLD)]
        b = [int(nxt()) for _ in range(K_OLD)]
        A_old.append(A); b_old.append(b)
    # sessions ignored by this tier

    single = [realize1(A_new, b_new, s, K_OLD, K_EXTRA, P) for s in range(M_NEW)]

    out = [str(M_OLD)]
    for t in range(M_OLD):
        found = None
        for s in range(M_NEW):
            A1, b1 = single[s]
            if A1 == A_old[t] and b1 == b_old[t]:
                found = s
                break
        if found is None:
            out.append("0")
        else:
            out.append("1 %d" % found)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
