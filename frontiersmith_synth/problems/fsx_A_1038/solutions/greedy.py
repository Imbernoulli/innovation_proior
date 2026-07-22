# TIER: greedy
# The "obvious" recipe: textbook phase-conjugate beamforming. Split the whole
# emitter budget K across the T harbors (one contiguous sub-array per harbor,
# each sub-array phase-conjugated toward its own bearing) to maximize gain at
# every target. This never reasons about the protected bearings at all, so it
# leaks whatever sidelobe energy phase-conjugation happens to produce there.
import sys, math


def phase_conj(i, j, N, P):
    ang = -2 * math.pi * i * j / N
    return round(ang / (2 * math.pi / P)) % P


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it)); P = int(next(it))
    T = int(next(it))
    targets = [int(next(it)) for _ in range(T)]
    Q = int(next(it))
    for _ in range(Q):
        next(it); next(it)

    K = min(K, N)
    base, rem = K // T, K % T
    chosen = []
    pos = 0
    for gi in range(T):
        size = base + (1 if gi < rem else 0)
        jt = targets[gi]
        for k in range(size):
            i = pos + k
            chosen.append((i, phase_conj(i, jt, N, P)))
        pos += size

    print(len(chosen))
    for (i, p) in chosen:
        print(i, p)


main()
