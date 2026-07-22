# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    U = int(next(it)); T = int(next(it))
    next(it); next(it)  # PEN, BUDGET -- unused
    C = []; R = []
    for _ in range(U):
        C.append(float(next(it))); R.append(float(next(it)))
        next(it); next(it)  # r_clean, duration -- unused
    demand = [float(next(it)) for _ in range(T)]

    # Structurally the right shape (never retrofit; index-order fill), but
    # the mid-horizon step is deliberately zeroed out -- a coverage
    # violation the checker must reject, not merely a low score.
    mid = T // 2
    out_lines = [" ".join(["0"] * U)]
    for t in range(T):
        row = [0.0] * U
        if t != mid:
            need = demand[t]
            for i in range(U):
                if need <= 1e-9:
                    break
                use = min(C[i], need)
                row[i] = use
                need -= use
        out_lines.append(" ".join("%.4f" % x for x in row))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
