# TIER: strong
# The insight: fidelity is judged on the OBSERVABLE projection only, and the engine's
# internal scratch coordinates are reset to 0 at the start of every OLD-message call. That
# means the composed effect of a SEQUENCE of new messages, projected back onto the
# observable coordinates, is itself an exact affine map of the observable input -- valid
# for every session, not just "close enough" on average. So instead of only trying direct
# 1:1 substitutes (greedy), search compositions of up to 2 new messages via matrix algebra
# and pick an EXACT match. This recovers a correct emulation for every old type, including
# the ones with no single-message counterpart, so the shim never corrupts its own
# carried-forward observable state and errors never cascade across a session.
import sys


def matmul(A, B, mod):
    r, k, c = len(A), len(B), len(B[0])
    out = [[0] * c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            if a == 0:
                continue
            Bt = B[t]
            row = out[i]
            for j in range(c):
                row[j] = (row[j] + a * Bt[j]) % mod
    return out


def matvec(A, v, mod):
    r = len(A); n = len(v)
    out = [0] * r
    for i in range(r):
        Ai = A[i]
        s = 0
        for j in range(n):
            s += Ai[j] * v[j]
        out[i] = s % mod
    return out


def vecadd(a, b, mod):
    return [(x + y) % mod for x, y in zip(a, b)]


def realize1(A_new, b_new, s, K_OLD):
    A1 = [row[:K_OLD] for row in A_new[s][:K_OLD]]
    b1 = b_new[s][:K_OLD]
    return A1, b1


def realize2(A_new, b_new, s1, s2, K_OLD, P):
    M1 = [row[:K_OLD] for row in A_new[s1]]        # K_NEW x K_OLD
    c1 = b_new[s1][:]                                # K_NEW
    A2M1 = matmul(A_new[s2], M1, P)                   # K_NEW x K_OLD
    A2c1 = matvec(A_new[s2], c1, P)
    b2 = vecadd(A2c1, b_new[s2], P)
    A_out = [row[:K_OLD] for row in A2M1[:K_OLD]]
    b_out = b2[:K_OLD]
    return A_out, b_out


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
    # sessions ignored by this tier -- the translation table is session-independent

    single = [realize1(A_new, b_new, s, K_OLD) for s in range(M_NEW)]

    out = [str(M_OLD)]
    for t in range(M_OLD):
        target = (A_old[t], b_old[t])
        best = None  # (length, seq)
        for s in range(M_NEW):
            if single[s] == target:
                best = (1, [s])
                break
        if best is None and L_MAX >= 2:
            for s1 in range(M_NEW):
                if best is not None:
                    break
                for s2 in range(M_NEW):
                    A2, b2 = realize2(A_new, b_new, s1, s2, K_OLD, P)
                    if A2 == A_old[t] and b2 == b_old[t]:
                        best = (2, [s1, s2])
                        break
        if best is None:
            out.append("0")
        else:
            _, seq = best
            out.append(str(len(seq)) + " " + " ".join(map(str, seq)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
