# TIER: greedy
# Obvious first instinct: ignore the trade network entirely and hand out each
# good PROPORTIONALLY to stated preference weight (largest-remainder rounding
# to hit the exact shipment size) -- the natural "give people what they say
# they want, in proportion" heuristic. It is a single pass: whatever a
# household's satiation cap clips away is simply left sitting with them
# (wasted, since they are already satiated) rather than being re-spread to
# someone who could still use it. It never models the community-local
# trading network at all, so a household whose community can never trade a
# good back to whoever values it more still receives -- and wastes -- a
# large share purely because it said it wants a lot of it.
import sys


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt()); G = int(nxt()); K = int(nxt()); R = int(nxt()); eps = int(nxt())
    cap = [int(nxt()) for _ in range(G)]
    S = [int(nxt()) for _ in range(G)]
    comm = [0] * N
    W = [[0] * G for _ in range(N)]
    for i in range(N):
        comm[i] = int(nxt())
        for g in range(G):
            W[i][g] = int(nxt())

    x = [[0] * G for _ in range(N)]
    for g in range(G):
        total_w = sum(W[i][g] for i in range(N))
        if total_w <= 0:
            base, rem = S[g] // N, S[g] % N
            for i in range(N):
                x[i][g] = base + (1 if i < rem else 0)
            continue
        shares = []
        floor_sum = 0
        for i in range(N):
            exact = S[g] * W[i][g] / total_w
            f = int(exact)
            shares.append((exact - f, i))
            x[i][g] = f
            floor_sum += f
        leftover = S[g] - floor_sum
        shares.sort(key=lambda t: (-t[0], t[1]))
        for k in range(leftover):
            x[shares[k][1]][g] += 1
        # single-pass cap clip: whatever a household's cap cuts away is simply
        # left sitting with them (permanently wasted for utility purposes),
        # not redistributed to anyone still short of their cap.

    print("\n".join(" ".join(map(str, row)) for row in x))


if __name__ == "__main__":
    main()
