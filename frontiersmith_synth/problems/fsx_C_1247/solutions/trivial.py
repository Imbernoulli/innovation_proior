# TIER: trivial
"""
Refresh EVERY row of every bank in a plain round-robin cycle, one refresh per slot,
period = Rb (that bank's own row count). Never looks at retention bounds (the generator
guarantees rho[b][i] >= Rb for all rows so this is always feasible) and never looks at the
access trace. This occupies 100% of every bank's slots, so it stalls every single access
request -- the checker's own reference baseline.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    T = int(nx())
    Bnum = int(nx())
    banks = []
    for _ in range(Bnum):
        Rb = int(nx())
        rho = [int(nx()) for _ in range(Rb)]
        banks.append(Rb)
    M = int(nx())
    for _ in range(M):
        nx(); nx(); nx()

    events = []
    for b, Rb in enumerate(banks):
        for t in range(T):
            row = t % Rb
            events.append((t, b, row))

    out = [str(len(events))]
    for (t, b, row) in events:
        out.append(f"{t} {b} {row}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
