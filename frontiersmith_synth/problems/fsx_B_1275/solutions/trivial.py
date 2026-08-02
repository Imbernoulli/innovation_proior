# TIER: trivial
import sys


def read_ints(tokens, k):
    return [int(next(tokens)) for _ in range(k)]


def main():
    toks = iter(sys.stdin.read().split())
    T, N, G = read_ints(toks, 3)
    V, P = read_ints(toks, 2)
    D = read_ints(toks, T)
    for _ in range(N):
        group, qualified, lead, qualcost, ntiers = read_ints(toks, 5)
        for _ in range(ntiers):
            read_ints(toks, 2)
    E = int(next(toks))
    for _ in range(E):
        read_ints(toks, 2)

    # Naive first instinct: single-source the only supplier that's already
    # qualified, and fully meet demand -- but place the order as a string of
    # small line items (never noticing that a single big order is what claims
    # the volume discount). Every line prices at the shallow base tier.
    chunk = 2
    lines = []
    for t in range(1, T + 1):
        Dt = D[t - 1]
        remaining = Dt
        while remaining > 0:
            q = min(chunk, remaining)
            lines.append((t, 0, q))
            remaining -= q

    out = ["0"]  # no qualification actions
    out.append(str(len(lines)))
    for t, i, q in lines:
        out.append(f"{t} {i} {q}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
