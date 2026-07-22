# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    U = int(next(it)); T = int(next(it))
    next(it); next(it)  # PEN, BUDGET -- unused
    C = []; R = []
    for _ in range(U):
        C.append(float(next(it))); R.append(float(next(it)))
        next(it); next(it)  # r_clean, duration -- unused (never retrofit)
    demand = [float(next(it)) for _ in range(T)]

    # Never retrofit; fixed INDEX-order fill to exactly meet demand each
    # step, no bonus dispatch. This is exactly the checker's own internal
    # naive baseline construction.
    out_lines = [" ".join(["0"] * U)]
    for t in range(T):
        need = demand[t]
        row = [0.0] * U
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
