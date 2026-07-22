# TIER: strong
"""Insight: build a MACHINE, not a MAP. Row 0 is seeded with T's own binary
digits (one dedicated tile per column position, k of them). Each COUNTING
row is obtained from the counting row below it by a temperature-2
COOPERATIVE ripple-borrow decrement: a middle column c can only place its
next tile once BOTH its south neighbor (its own previous bit) and its west
neighbor (the borrow rippling in from column c-1) agree -- a single
strength-1 glue is not enough at temperature 2.

The leftmost (least-significant) column has no lateral neighbor of its
own, so by itself it could keep decrementing forever, oblivious to
whether the row it just finished was already all-zero. Between every
pair of counting rows we therefore insert one reusable RELAY row that
sweeps the OPPOSITE direction (most-significant column first, one column
at a time, west): each relay cell cooperatively reads its own bit
(south) and an incoming running-OR flag (east) and passes on their OR
(west) while faithfully re-emitting the actual bit value upward. Only
when the relay reaches the least-significant column does the full row's
OR become known -- and that column simply has no tile type for the one
combination "my own bit is 0 and everything to my right was also 0",
i.e. the row below was genuinely all-zero. Omitting exactly that one
combination is the entire halt condition: the relay silently fails to
finish, so no cell of the counting row above it ever gets built, at
every column, all at once.

Every relay/decrement tile type is reused at EVERY row and EVERY
applicable column, so the total tile-type count is O(bit_length(T)) =
O(log T), not O(T).
"""
import sys


def main():
    T = int(sys.stdin.read().split()[0])
    k = max(1, T.bit_length())
    Tbits = [(T >> c) & 1 for c in range(k)]

    _next = [1]

    def new_label():
        v = _next[0]
        _next[0] += 1
        return v

    BIT = [[new_label(), new_label()] for _ in range(k)]          # BIT[c][v]
    BORROW = [[new_label(), new_label()] for _ in range(k - 1)] if k > 1 else []
    CHAIN = [new_label() for _ in range(k - 1)] if k > 1 else []   # row-0 column hand-off
    ORCH = [new_label(), new_label()] if k > 1 else None            # relay running-OR flag

    def down_st(c):  # strength a counting row's SOUTH-match uses reading the relay below it
        return 2 if c == 0 else 1

    def up_st(c):    # strength a counting row's NORTH-emit uses feeding the relay above it
        return 2 if c == k - 1 else 1

    types = []  # rows of [Nl,Ns,El,Es,Sl,Ss,Wl,Ws]

    def emit(N=(0, 0), E=(0, 0), S=(0, 0), W=(0, 0)):
        types.append([N[0], N[1], E[0], E[1], S[0], S[1], W[0], W[1]])
        return len(types)  # 1-based id

    # ---- row 0: seed the columns with T's own bits ----
    n0 = (BIT[0][Tbits[0]], up_st(0))
    if k > 1:
        seed_id = emit(N=n0, E=(CHAIN[0], 2))
    else:
        seed_id = emit(N=n0)
    for c in range(1, k):
        n_c = (BIT[c][Tbits[c]], up_st(c))
        w_c = (CHAIN[c - 1], 2)
        if c < k - 1:
            emit(N=n_c, E=(CHAIN[c], 2), W=w_c)
        else:
            emit(N=n_c, W=w_c)

    if k == 1:
        # a single column locally detects "already at zero" -> no relay needed
        if T >= 1:
            emit(N=(BIT[0][0], 2), S=(BIT[0][1], 2))
        out = [str(len(types))]
        for row in types:
            out.append(" ".join(map(str, row)))
        out.append(str(seed_id))
        sys.stdout.write("\n".join(out) + "\n")
        return

    # ---- ripple-borrow decrement machinery (reused at EVERY counting row) ----
    # column 0 (LSB): borrow_in is always 1 (we always subtract exactly 1)
    emit(N=(BIT[0][0], up_st(0)), S=(BIT[0][1], down_st(0)), E=(BORROW[0][0], 1))  # old=1->new=0
    emit(N=(BIT[0][1], up_st(0)), S=(BIT[0][0], down_st(0)), E=(BORROW[0][1], 1))  # old=0->new=1

    for c in range(1, k - 1):  # interior columns
        for old in (0, 1):
            for bin_ in (0, 1):
                new = old ^ bin_
                bout = (1 - old) & bin_
                emit(N=(BIT[c][new], up_st(c)), E=(BORROW[c][bout], 1),
                     S=(BIT[c][old], down_st(c)), W=(BORROW[c - 1][bin_], 1))

    # most-significant column: kept as a redundant safety omission (the relay's
    # own OR-check below is what actually prevents an all-zero row from ever
    # reaching this computation in the first place).
    c = k - 1
    for (old, bin_) in ((0, 0), (1, 0), (1, 1)):
        new = old ^ bin_
        emit(N=(BIT[c][new], up_st(c)), S=(BIT[c][old], down_st(c)), W=(BORROW[c - 1][bin_], 1))

    # ---- relay row machinery (reused at EVERY relay row) ----
    # most-significant column starts the relay: always continues, seeds the
    # running-OR flag with its own bit, re-emits its value upward.
    c = k - 1
    for v in (0, 1):
        emit(N=(BIT[c][v], down_st(c)), S=(BIT[c][v], up_st(c)), W=(ORCH[v], 1))

    # interior relay cells: cooperate south (own bit) + east (running OR so
    # far), always continue, pass the OR onward west and the bit value north.
    for c in range(1, k - 1):
        for v in (0, 1):
            for orin in (0, 1):
                orout = v | orin
                emit(N=(BIT[c][v], down_st(c)), E=(ORCH[orin], 1),
                     S=(BIT[c][v], up_st(c)), W=(ORCH[orout], 1))

    # least-significant column: the ONE omitted combination (own bit 0, no OR
    # from the right) means "this whole row was all-zero" -- exactly there we
    # emit nothing, so the counting row above never gets a south neighbor at
    # ANY column, at once. This is the entire halt condition.
    c = 0
    for (v, orin) in ((1, 0), (1, 1), (0, 1)):
        emit(N=(BIT[c][v], down_st(c)), S=(BIT[c][v], up_st(c)), E=(ORCH[orin], 1))

    out = [str(len(types))]
    for row in types:
        out.append(" ".join(map(str, row)))
    out.append(str(seed_id))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
