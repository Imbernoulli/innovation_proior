# TIER: greedy
"""
The obvious "safe" recipe: for each bank, compute the bank-wide WORST-CASE retention
(the minimum rho over all its rows) and refresh EVERY row that often, round-robin, at the
earliest legal slot -- i.e. treat every row as if it were as weak as the weakest row in its
bank. This is correct everywhere (uses the true per-bank worst case, so it never violates
retention) and it DOES exploit bank-parallelism (each bank gets its own worst-case interval,
not a single global one). But it ignores per-row retention variation (strong rows are
refreshed just as often as weak ones) and it never looks at the access trace, so it cannot
avoid slots the access trace is hammering.
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
        banks.append((Rb, rho))
    M = int(nx())
    for _ in range(M):
        nx(); nx(); nx()

    events = []
    for b, (Rb, rho) in enumerate(banks):
        m_b = min(rho)  # per-bank worst-case (safe) interval
        # round-robin: row i fires at i, i+m_b, i+2*m_b, ... (fits since m_b >= Rb)
        for i in range(Rb):
            t = i
            while t < T:
                events.append((t, b, i))
                t += m_b

    out = [str(len(events))]
    for (t, b, row) in events:
        out.append(f"{t} {b} {row}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
