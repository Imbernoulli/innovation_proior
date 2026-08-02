# TIER: greedy
"""The obvious first approach: at each timestep independently, greedily
explain the CURRENT aggregate reading by "subtracting" appliance power
levels (largest first) -- a per-instant subset-sum recipe, exactly the
"subtract mean/level powers" trap. It never looks at which timestep the
aggregate actually changed, nor at how long an appliance has already
dwelt in its current state, so it cannot tell apart two appliances that
share a power level but differ in legal dwell timing, and it has no
principled way to split a value that is the SUM of several concurrently-ON
appliances. A simple debounce pass then repairs the raw per-timestep
guesses into a legal sequence (so the output is always feasible), but the
underlying appliance-identity guess is never revisited."""
import sys


def raw_guess(aggregate, archs):
    T = len(aggregate)
    A = len(archs)
    order = sorted(range(A), key=lambda i: -archs[i][0])  # descending power
    raw = [[0] * T for _ in range(A)]
    for t in range(T):
        rem = aggregate[t]
        for i in order:
            P = archs[i][0]
            if rem >= P:
                raw[i][t] = 1
                rem -= P
    return raw


def repair(raw, mon, mxon, moff, mxoff):
    T = len(raw)
    out = []
    state = 0
    elapsed = 0
    for t in range(T):
        want = raw[t]
        maxd = mxoff if state == 0 else mxon
        mind = moff if state == 0 else mon
        if elapsed >= maxd:
            state = 1 - state
            elapsed = 0
        elif want != state and elapsed >= mind:
            state = 1 - state
            elapsed = 0
        out.append(state)
        elapsed += 1
    return out


def main():
    data = sys.stdin.read().split()
    ptr = 0
    T = int(data[ptr]); A = int(data[ptr + 1]); ptr += 2
    ptr += 2  # wT wA unused by this recipe
    archs = []
    for _ in range(A):
        P, mon, mxon, moff, mxoff = (int(x) for x in data[ptr:ptr + 5]); ptr += 5
        archs.append((P, mon, mxon, moff, mxoff))
    aggregate = [int(x) for x in data[ptr:ptr + T]]; ptr += T

    raw = raw_guess(aggregate, archs)

    out = [str(A)]
    for i, (P, mon, mxon, moff, mxoff) in enumerate(archs):
        seq = repair(raw[i], mon, mxon, moff, mxoff)
        out.append(" ".join(str(v) for v in seq))
    print("\n".join(out))


if __name__ == "__main__":
    main()
