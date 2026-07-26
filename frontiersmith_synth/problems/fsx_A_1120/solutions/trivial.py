# TIER: trivial
# Naive fixed point: assume physical bay k already holds carousel k (i.e. book
# id's home bay id//M is wherever it happens to be sitting right now -- never
# read the stamp to jump straight there).  For each book in id order, spin its
# assumed home bay's shelf forward hoping it turns up; if a full revolution
# proves that guess wrong, nudge the whole hall forward once and try the next
# guess.  Always valid and terminating, but every wrong guess wastes a full
# M-step revolution -- reproduces the evaluator's own weak reference exactly,
# so this scores ~0.1 on every instance.
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

    for i in range(N):
        target_bay = i // M
        hall_tries = 0
        while state[i] != i:
            found = False
            for _ in range(M):
                if state[i] == i:
                    found = True
                    break
                state = apply_move(state, K, M, f"S{target_bay}+")
                moves.append(f"S{target_bay}+")
            if found:
                break
            state = apply_move(state, K, M, "H+")
            moves.append("H+")
            hall_tries += 1
            if hall_tries > K + 2:
                break

    print(json.dumps({"moves": moves}))


if __name__ == "__main__":
    main()
