#!/usr/bin/env python3
# gen.py <testId>  -> prints ONE retrosynthesis-route instance to stdout.
#
# Planted structure (deterministic, testId only selects the ladder row):
#   * A LINEAR chain from a raw material RM_LIN to the target, Dlin reactions deep.
#     Every "ordinary" chain step actually offers TWO reactions with the SAME yield
#     but different cost (a cheap one, a pricier one with a smaller reaction id) --
#     a pure cost tie-break that has nothing to do with yield.
#   * One position in the chain (k == mid) is a genuine protecting-group fork:
#     a 1-step "shortcut" reaction (low id, low yield, no protection) versus a
#     3-step protect -> react -> deprotect path (higher ids, high yield per step).
#   * A CONVERGENT alternative: two independently-built branches (from RM_A, RM_B)
#     that couple in a single final reaction into the target. This route uses
#     MORE total reactions than the linear route but multiplies far fewer yield
#     factors onto its expensive early material, because each branch is shallow.
#   * The target itself has exactly two producing reactions: the linear-final
#     reaction (smaller id) and the convergent-coupling reaction (larger id).
#
# All reaction ids are assigned so that "always take the smallest-id reaction
# available for the molecule you need" (the checker's own naive baseline, and
# solutions/trivial.py) reproduces the linear+shortcut+expensive-tiebreak route,
# and "always take the fewest-total-reaction-steps route, tie-broken by that same
# naive per-step cost" (solutions/greedy.py) reproduces the linear+shortcut route
# with the cheap tie-break -- both miss the convergent + protecting-path optimum
# that solutions/strong.py finds via a bottom-up minimum-delivered-cost DP.
import sys

# ------------------------------------------------------------------ ladder
# testId -> Dlin (total reactions in the linear route, chain-build steps + final)
LADDER = {1: 4, 2: 4, 3: 5, 4: 5, 5: 6, 6: 6, 7: 7, 8: 7, 9: 8, 10: 8}

TARGET = 0
RM_LIN = 1
RM_A = 2
RM_B = 3

Y_CHAIN = 85
C_CHAIN_X = 6          # expensive (lower id) chain-step option
C_CHAIN_Y = 2           # cheap (higher id) chain-step option -- same yield
Y_SHORT = 22
C_SHORT = 3
Y_PROT = 90
C_PROT = 2
C_INNER = 3
Y_FINAL_LIN = 85
C_FINAL_LIN = 4
Y_BRANCH = 88
C_BRANCH = 2
Y_CONV = 80
C_CONV = 5
RM_LIN_COST = 3
RM_A_COST = 2
RM_B_COST = 2


def build_instance(tid):
    Dlin = LADDER[tid]
    mid = max(2, Dlin // 2)
    dA = mid + 1
    dB = mid + 1

    reactions = []  # (id, output, [inputs], yield_pct, cost)
    rid = 0

    def L(k):
        return 10 + (k - 1)   # molecule id for chain position k (1 .. Dlin-1)

    prev = RM_LIN
    for k in range(1, Dlin):  # build L(1) .. L(Dlin-1)
        out = L(k)
        if k == mid:
            # shortcut (low id, low yield, 1 step)
            reactions.append((rid, out, [prev], Y_SHORT, C_SHORT)); rid += 1
            # protect -> inner -> deprotect (higher ids, 3 steps, high per-step yield)
            prot = 500
            prot2 = 501
            reactions.append((rid, prot, [prev], Y_PROT, C_PROT)); rid += 1
            reactions.append((rid, prot2, [prot], Y_PROT, C_INNER)); rid += 1
            reactions.append((rid, out, [prot2], Y_PROT, C_PROT)); rid += 1
        else:
            # expensive option first (smaller id), cheap option second (larger id)
            reactions.append((rid, out, [prev], Y_CHAIN, C_CHAIN_X)); rid += 1
            reactions.append((rid, out, [prev], Y_CHAIN, C_CHAIN_Y)); rid += 1
        prev = out

    lin_last_id = rid
    reactions.append((rid, TARGET, [prev], Y_FINAL_LIN, C_FINAL_LIN)); rid += 1

    def A(j):
        return 100 + (j - 1)

    def B(j):
        return 200 + (j - 1)

    prevA = RM_A
    for j in range(1, dA + 1):
        reactions.append((rid, A(j), [prevA], Y_BRANCH, C_BRANCH)); rid += 1
        prevA = A(j)

    prevB = RM_B
    for j in range(1, dB + 1):
        reactions.append((rid, B(j), [prevB], Y_BRANCH, C_BRANCH)); rid += 1
        prevB = B(j)

    conv_id = rid
    reactions.append((rid, TARGET, [prevA, prevB], Y_CONV, C_CONV)); rid += 1

    purchasable = {RM_LIN: RM_LIN_COST, RM_A: RM_A_COST, RM_B: RM_B_COST}

    max_id = 0
    for (_, out, ins, _, _) in reactions:
        max_id = max(max_id, out, *ins)
    N = max_id + 1
    M = len(reactions)
    P = len(purchasable)
    return N, M, TARGET, P, purchasable, reactions


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    if tid not in LADDER:
        tid = ((tid - 1) % 10) + 1
    N, M, T, P, purchasable, reactions = build_instance(tid)

    out = []
    out.append("%d %d %d %d" % (N, M, T, P))
    for mol in sorted(purchasable):
        out.append("%d %d" % (mol, purchasable[mol]))
    for (rid, output, ins, ypct, cost) in reactions:
        k = len(ins)
        out.append("%d %d %d %s %d %d" % (rid, output, k, " ".join(map(str, ins)), ypct, cost))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
