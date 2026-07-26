# TIER: greedy
# Reads the stamp: book id's true home is carousel id//M, pocket id%M -- so
# this DOES discover which physical bay currently holds book 0's carousel
# (look at state[0], its home bay is state[0]//M, so the hall needs
# state[0]//M more forward turns) and then, per bay, discovers the shelf
# offset (state[bay*M] % M).  That is real structure discovery, unlike the
# trivial tier.  But it always spins FORWARD only -- it never checks whether
# going backward would be shorter, at either level -- so it is a "found the
# recipe, didn't polish it" solution: no wasted revolutions, but each of the
# K+1 independent degrees of freedom is fixed by whichever of its two
# directions the candidate happens to try first (here: always "+").
import sys, json


def apply_move(state, K, M, mv):
    ns = state[:]
    if mv == "H+":
        for k in range(K):
            bn, bo = ((k + 1) % K) * M, k * M
            for p in range(M):
                ns[bn + p] = state[bo + p]
    else:
        k = int(mv[1:-1])
        base = k * M
        for p in range(M):
            ns[base + (p + 1) % M] = state[base + p]
    return ns


def main():
    inst = json.load(sys.stdin)
    K, M, N = inst["K"], inst["M"], inst["N"]
    state = list(inst["state"])
    moves = []

    # hall level: how many forward turns until book 0's home carousel is at bay 0
    X = state[0] // M
    for _ in range(X):
        state = apply_move(state, K, M, "H+")
        moves.append("H+")

    # shelf level, per bay, forward only
    for k in range(K):
        loc0 = state[k * M] % M
        for _ in range(loc0):
            state = apply_move(state, K, M, f"S{k}+")
            moves.append(f"S{k}+")

    print(json.dumps({"moves": moves}))


if __name__ == "__main__":
    main()
