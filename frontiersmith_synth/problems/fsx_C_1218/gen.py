#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Family: wire-protocol-transcode.  A shim must translate messages from an OLD wire
protocol into a sequence of messages of a NEW engine's protocol, so that a client
that only ever observes the OLD protocol's session state cannot tell the difference.

State model (all arithmetic mod P):
  - OLD protocol state = a vector of K_OLD integers ("observable" fields).
  - NEW engine state   = a vector of K_NEW = K_OLD + K_EXTRA integers: the first
    K_OLD coordinates are the OBSERVABLE fields (aligned 1:1 with the OLD protocol's
    fields), the remaining K_EXTRA are INTERNAL scratch that no client ever sees.
  - Every OLD message type t has a fixed affine transform (A_t, b_t) on the K_OLD
    observable vector: obs' = A_t . obs + b_t  (mod P).
  - Every NEW message type s has a fixed affine transform (A'_s, b'_s) on the full
    K_NEW vector: v' = A'_s . v + b'_s (mod P).
  - Whenever the shim begins translating ONE old message, the engine's internal
    scratch coordinates are (by protocol contract) reset to 0 for that call; the
    observable coordinates carry over from whatever the session currently holds.

testId 1..10 is a difficulty ladder over session count / length. Everything is
seeded from testId only, so generation is bit-for-bit reproducible. gen.py prints
ONLY the instance (protocol specs + client sessions) -- never which old types are
"direct" vs "gap", nor which new-type sequence realizes a gap type: the solver must
discover that by searching compositions of the given affine maps.
"""
import random
import sys

K_OLD = 3
K_EXTRA = 2
K_NEW = K_OLD + K_EXTRA
P = 1_000_000_007
L_MAX = 2                      # max NEW messages a translation entry may emit
M_OLD = 7                      # old type 0 = identity/handshake, 1-4 direct, 5-6 gap
M_NEW = 7                      # new type 0 = identity, 1-6 generic
CPX_NUM, CPX_DEN = 1, 2        # complexity cost per extra emitted message (1/2)

# session-size ladder: (num_sessions, session_length) by testId (1-indexed)
LADDER = [
    (6, 4), (8, 5), (10, 6), (10, 7), (12, 8),
    (12, 9), (14, 10), (14, 11), (16, 12), (16, 12),
]


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def rand_matrix(rng, rows, cols, mod):
    return [[rng.randrange(mod) for _ in range(cols)] for _ in range(rows)]


def rand_vec(rng, n, mod):
    return [rng.randrange(mod) for _ in range(n)]


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
    r = len(A)
    n = len(v)
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


def top_block(A, rows, cols):
    return [row[:cols] for row in A[:rows]]


def top_vec(v, n):
    return v[:n]


def realize1(A_new, b_new, s):
    """Single new-type s applied from (obs, 0,...,0): returns (A,b) on K_OLD dims."""
    A_full = top_block(A_new[s], K_NEW, K_OLD)  # K_NEW x K_OLD (cols that matter)
    A1 = top_block(A_full, K_OLD, K_OLD)
    b1 = top_vec(b_new[s], K_OLD)
    return A1, b1


def realize2(A_new, b_new, s1, s2):
    """Composition new-type s1 then s2, from (obs, 0,...,0): returns (A,b) on K_OLD dims."""
    M1 = top_block(A_new[s1], K_NEW, K_OLD)      # K_NEW x K_OLD
    c1 = b_new[s1][:]                             # K_NEW
    A2M1 = matmul(A_new[s2], M1, P)                # K_NEW x K_OLD
    A2c1 = matvec(A_new[s2], c1, P)
    b2 = vecadd(A2c1, b_new[s2], P)
    A_out = top_block(A2M1, K_OLD, K_OLD)
    b_out = top_vec(b2, K_OLD)
    return A_out, b_out


def eq(A1, b1, A2, b2):
    return A1 == A2 and b1 == b2


def build_new_types(rng):
    A_new = [identity(K_NEW)]
    b_new = [[0] * K_NEW]
    for _ in range(1, M_NEW):
        A_new.append(rand_matrix(rng, K_NEW, K_NEW, P))
        b_new.append(rand_vec(rng, K_NEW, P))
    return A_new, b_new


def build_old_types(rng, A_new, b_new):
    A_old = [identity(K_OLD)]
    b_old = [[0] * K_OLD]

    # 4 DIRECT types: each realized by exactly one distinct new type (excluding 0).
    direct_sources = rng.sample(range(1, M_NEW), 4)
    for s in direct_sources:
        A1, b1 = realize1(A_new, b_new, s)
        A_old.append(A1)
        b_old.append(b1)

    # 2 GAP types: each realized only by a length-2 composition; verified that no
    # single new type (including identity) reproduces it.
    for _ in range(2):
        tries = 0
        while True:
            s1 = rng.randrange(1, M_NEW)
            s2 = rng.randrange(1, M_NEW)
            A2, b2 = realize2(A_new, b_new, s1, s2)
            collision = False
            for s in range(M_NEW):
                A1, b1 = realize1(A_new, b_new, s)
                if eq(A1, b1, A2, b2):
                    collision = True
                    break
            tries += 1
            if not collision or tries > 500:
                break
        A_old.append(A2)
        b_old.append(b2)

    return A_old, b_old


def gen_sessions(rng, num_sessions, length):
    # weighted pool over old-type ids 0..6: identity(0)x1, direct(1-4)x1 each, gap(5,6)x2 each
    pool = [0, 1, 2, 3, 4, 5, 5, 6, 6]
    sessions = []
    for _ in range(num_sessions):
        init = rand_vec(rng, K_OLD, P)
        seq = [0]  # forced handshake first
        for _ in range(length - 1):
            seq.append(pool[rng.randrange(len(pool))])
        sessions.append((init, seq))
    return sessions


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    idx = max(1, min(10, test_id)) - 1
    num_sessions, length = LADDER[idx]

    rng = random.Random(900001 + 97 * test_id)
    A_new, b_new = build_new_types(rng)
    A_old, b_old = build_old_types(rng, A_new, b_new)
    sessions = gen_sessions(rng, num_sessions, length)

    out = []
    out.append(f"{K_OLD} {K_NEW} {M_OLD} {M_NEW} {P} {L_MAX} {CPX_NUM} {CPX_DEN}")
    out.append("NEWTYPES")
    for s in range(M_NEW):
        for row in A_new[s]:
            out.append(" ".join(map(str, row)))
        out.append(" ".join(map(str, b_new[s])))
    out.append("OLDTYPES")
    for t in range(M_OLD):
        for row in A_old[t]:
            out.append(" ".join(map(str, row)))
        out.append(" ".join(map(str, b_old[t])))
    out.append("SESSIONS")
    out.append(str(len(sessions)))
    for init, seq in sessions:
        out.append(" ".join(map(str, init)))
        out.append(str(len(seq)))
        out.append(" ".join(map(str, seq)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
