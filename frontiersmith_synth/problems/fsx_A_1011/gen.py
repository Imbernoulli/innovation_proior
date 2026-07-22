import sys, math, random

# ---- Family: interleaved-reheat-temper-shop | theme: blacksmith at one forge ----
# W workpieces each need 3 operations, in order, performed while the piece's
# temperature sits inside a fixed band (HOT -> MID -> LOW). Pieces cool passively
# by Newton's law whenever the forge is not actively holding them; a reheat resets
# a piece to REHEAT_TEMP but multiplies its eventual quality by PENALTY. Difficulty
# ladder: more workpieces and a wider spread of cooling rates as testId grows.

Tamb = 20.0
BANDS = [(800.0, 900.0), (400.0, 500.0), (150.0, 250.0)]   # HOT, MID, LOW
REHEAT_TEMP = 900.0
REHEAT_DUR = 6
OP_DUR = 2
PENALTY = 0.65

W_LIST = [3, 3, 4, 4, 5, 6, 6, 7, 7, 8]


def cool_time_to(T_from, T_to, k):
    """Time (>=0) for Newton cooling to fall from T_from to T_to (T_to < T_from)."""
    if T_from <= T_to:
        return 0.0
    return math.log((T_from - Tamb) / (T_to - Tamb)) / k


def params_for_testid(tid):
    W = W_LIST[(tid - 1) % len(W_LIST)]
    k_lo = 0.005
    k_hi = 0.009 + 0.0015 * tid
    seed = 900000 + tid * 7919
    return W, k_lo, k_hi, seed


def make_pieces(tid):
    W, k_lo, k_hi, seed = params_for_testid(tid)
    rng = random.Random(seed)
    pieces = []
    for _ in range(W):
        T0 = rng.uniform(850.0, 900.0)
        k = rng.uniform(k_lo, k_hi)
        vs = [rng.randint(4, 10) for _ in range(3)]
        pieces.append((T0, k, vs))
    return pieces


def safe_baseline_time(pieces):
    """Time used by the 'always reheat before every operation' construction --
    the same one the checker uses as its scoring baseline B. Sized generously so
    HORIZON always has slack for it (and hence for any less wasteful schedule)."""
    t = 0.0
    for (T0, k, vs) in pieces:
        for (lo, hi) in BANDS:
            t += REHEAT_DUR
            target = (lo + hi) / 2.0
            t += cool_time_to(REHEAT_TEMP, target, k)
            t += OP_DUR
    return t


def main():
    tid = int(sys.argv[1])
    pieces = make_pieces(tid)
    W = len(pieces)
    base_t = safe_baseline_time(pieces)
    horizon = int(math.ceil(base_t * 1.10)) + 10

    out = []
    out.append(str(W))
    out.append("%.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (
        Tamb, BANDS[0][0], BANDS[0][1], BANDS[1][0], BANDS[1][1], BANDS[2][0], BANDS[2][1]))
    out.append("%.6f %d %d %.6f %d" % (REHEAT_TEMP, REHEAT_DUR, OP_DUR, PENALTY, horizon))
    for (T0, k, vs) in pieces:
        out.append("%.6f %.6f %d %d %d" % (T0, k, vs[0], vs[1], vs[2]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
