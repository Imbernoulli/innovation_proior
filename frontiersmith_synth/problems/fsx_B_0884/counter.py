# counter.py -- Format D checker for fsx_B_0884 (nonlinear-propagation-flop-schedule).
#
# Input (<in>):
#   N K CAP
#   K lines, each: W[a][0] ... W[a][K-1]
#   c_0 ... c_{N-1}                      (community id of each of the N agents)
#
#   Dynamics (Jacobi / synchronous), for an UNKNOWN per-community bias vector
#   beta[0..K-1] with 1 <= beta[a] <= CAP:
#     x_i^(0) = 0
#     x_i^(t+1) = min(CAP, beta[c_i] + sum_{j != i} W[c_i][c_j] * x_j^(t))
#   All agents in the same community share one trajectory (community-symmetric
#   start + community-only-dependent update), and the sequence is coordinate-
#   wise non-decreasing (weights/biases >= 0) and bounded by CAP -> reaches an
#   exact integer fixed point.  PROOF that this happens within K*CAP rounds,
#   for ANY beta in [0,CAP]^K: each round that has not yet converged strictly
#   increases at least one coordinate (monotone + not-yet-equal => some
#   coordinate rose); each individual coordinate, being a non-decreasing
#   integer sequence confined to [0, CAP], can strictly rise at most CAP
#   times over the whole run. So total rounds <= K*CAP. This bound does NOT
#   depend on beta's actual value -- only on K, CAP (and the fixed W is only
#   used through boundedness, also beta-independent) -- which is exactly what
#   lets a solver build ONE fixed circuit that is correct for every beta.
#
# beta is deliberately NOT given in <in>. Instead it is an UNKNOWN that this
# checker substitutes at several independently-drawn test points (K implicit
# INPUT nodes, indices 0..K-1, exactly like the free variables of a
# polynomial-identity SLP checker) and verifies the submission at every one
# of them. This is what makes "hardcode the final numeric answer as a const"
# fail: a hardcoded circuit is only right for the one beta it was tuned to.
#
# Output (<out>): a straight-line program (SLP) over the integers.
#     L
#     L lines, each one of:
#         const v            # a new node holding the integer constant v  (FREE)
#         add i j             # node[i] + node[j]        (cost 1)
#         sub i j             # node[i] - node[j]        (cost 1)
#         mul i j             # node[i] * node[j]        (cost 1)
#         min i j             # min(node[i], node[j])    (cost 1)
#     out q_0 ... q_{N-1}
#   Nodes 0..K-1 are the IMPLICIT beta inputs (beta_0..beta_{K-1}); they are
#   NOT emitted by the participant.  Instruction t (0-based) defines node
#   K+t and may reference only earlier nodes (index < K+t) -- a DAG.  `out`
#   names, for every agent, the node that must equal its converged value for
#   EVERY substituted beta.  No trailing tokens are allowed after `out`'s N
#   indices (strict format).
#
# COST MODEL: cost = number of add/sub/mul/min instructions. `const` and the
# implicit beta inputs are free.
#
# Scoring (minimize cost):
#   1) STRICT feasibility: parse + range checks (matching the stated domain);
#      reject non-integer/garbage, bad references, wrong out-line length,
#      trailing tokens, or any simulated value leaving a generous safe
#      integer range (blocks exponential blow-up DoS such as repeated
#      self-squaring).
#   2) EXACT equivalence at T independently-drawn beta test points (each
#      beta[a] uniform in [1, CAP], seeded deterministically from the
#      instance): simulate the submitted SLP over Python big ints for each
#      beta, compare every out[i] to the checker's own reference fixed
#      point. Any mismatch at any trial -> Ratio 0.0.
#   3) Baseline B = analytic op count of the canonical "simulate every agent,
#      every individual incoming edge, every round, for K*CAP rounds (the
#      round bound any beta-agnostic circuit must use)" construction.
#      Ratio = min(1.0, 0.1 * B / cost).
import sys, random

VAL_BOUND = 10 ** 15   # generous vs. true values (<= CAP <= 50); blocks blow-up DoS
MAX_L = 2_000_000
N_MAX, K_MAX, CAP_MAX, W_MAX = 200, 200, 50, 3
T_TRIALS = 10


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def mult(a, b, n):
    return n[b] if b != a else max(0, n[a] - 1)


def main():
    try:
        ind = open(sys.argv[1]).read().split()
    except Exception:
        fail("io-in")
    try:
        outd = open(sys.argv[2]).read().split()
    except Exception:
        fail("io-out")

    it = iter(ind)
    try:
        N = int(next(it)); K = int(next(it)); CAP = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= K <= N <= N_MAX) or K > K_MAX:
        fail("bad N/K")
    if not (1 <= CAP <= CAP_MAX):
        fail("bad CAP")
    try:
        W = [[int(next(it)) for _ in range(K)] for _ in range(K)]
        c = [int(next(it)) for _ in range(N)]
    except Exception:
        fail("truncated instance")
    if any(not (0 <= w <= W_MAX) for row in W for w in row):
        fail("bad W")
    if any(not (0 <= ci < K) for ci in c):
        fail("bad community id")

    n = [0] * K
    for ci in c:
        n[ci] += 1

    nz = [[b for b in range(K) if W[a][b] != 0 and mult(a, b, n) != 0] for a in range(K)]
    R_max = K * CAP + 2   # beta-independent worst-case round bound, see proof above

    # ---- baseline B: naive per-agent / per-individual-edge / per-round simulation,
    #      run for R_max rounds (the round budget any beta-agnostic circuit needs) ----
    deg_node = [0] * K
    for a in range(K):
        for b in nz[a]:
            deg_node[a] += mult(a, b, n)
    B = 0
    for a in range(K):
        B += n[a] * (2 * deg_node[a] + 1)
    B *= R_max
    if B <= 0:
        fail("degenerate instance")

    # ---- parse participant SLP (structural checks only; values come later) ----
    if not outd:
        fail("empty output")
    oit = iter(outd)
    try:
        L = int(next(oit))
    except Exception:
        fail("bad L")
    if L < 0 or L > MAX_L:
        fail("L out of range")

    ops = []            # (kind, a, b): kind 0=const(value) 1=add 2=sub 3=mul 4=min
    cost = 0
    OPCODE = {"add": 1, "sub": 2, "mul": 3, "min": 4}
    try:
        for t in range(L):
            op = next(oit)
            cur = K + t
            if op == "const":
                v = int(next(oit))
                if not (-VAL_BOUND <= v <= VAL_BOUND):
                    fail("const out of range")
                ops.append((0, v, 0))
            elif op in OPCODE:
                i = int(next(oit)); j = int(next(oit))
                if not (0 <= i < cur and 0 <= j < cur):
                    fail("bad node reference")
                cost += 1
                ops.append((OPCODE[op], i, j))
            else:
                fail("bad opcode")
        if next(oit) != "out":
            fail("missing out line")
        outs = [int(next(oit)) for _ in range(N)]
    except SystemExit:
        raise
    except StopIteration:
        fail("truncated output")
    except ValueError:
        fail("non-integer token")
    except Exception:
        fail("parse error")

    if next(oit, None) is not None:
        fail("trailing tokens after out line")

    for o in outs:
        if not (0 <= o < K + L):
            fail("bad out index")

    # ---- T independently-drawn beta trials, deterministic from the instance ----
    seedmix = 0x2545F4914F6CDD1D
    seedmix = (seedmix * 1000003 + N) & ((1 << 64) - 1)
    seedmix = (seedmix * 1000003 + K) & ((1 << 64) - 1)
    seedmix = (seedmix * 1000003 + CAP) & ((1 << 64) - 1)
    for row in W:
        for w in row:
            seedmix = (seedmix * 1000003 + w + 1) & ((1 << 64) - 1)
    for ci in c:
        seedmix = (seedmix * 1000003 + ci + 1) & ((1 << 64) - 1)
    rng = random.Random(seedmix)

    for _trial in range(T_TRIALS):
        beta = [rng.randint(1, CAP) for _ in range(K)]

        # reference fixed point for this beta (K-dimensional, exact integers)
        y = [0] * K
        rounds = 0
        while True:
            ny = []
            for a in range(K):
                s = beta[a]
                for b in nz[a]:
                    s += W[a][b] * mult(a, b, n) * y[b]
                ny.append(s if s < CAP else CAP)
            if ny == y:
                break
            y = ny
            rounds += 1
            if rounds > R_max + 50:
                fail("reference did not converge (generator bug)")

        # simulate the submitted SLP with beta substituted into nodes 0..K-1
        vals = list(beta)
        for kind, a, b in ops:
            if kind == 0:
                r = a
            else:
                vi, vj = vals[a], vals[b]
                if kind == 1:
                    r = vi + vj
                elif kind == 2:
                    r = vi - vj
                elif kind == 3:
                    r = vi * vj
                else:
                    r = vi if vi < vj else vj
            if not (-VAL_BOUND <= r <= VAL_BOUND):
                fail("value out of range (overflow/blow-up)")
            vals.append(r)

        for i in range(N):
            if vals[outs[i]] != y[c[i]]:
                fail("reconstruction mismatch at agent %d (trial %d)" % (i, _trial))

    ratio = min(1.0, 0.1 * B / max(1, cost))
    print("N=%d K=%d R_max=%d B=%d cost=%d Ratio: %.6f" % (N, K, R_max, B, cost, ratio))


if __name__ == "__main__":
    main()
