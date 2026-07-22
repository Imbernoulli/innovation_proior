import sys, random

# gen.py <testId>  -- prints ONE "one ladder for many exponents" instance.
#
# We ship a batch of k target integers, each exactly 60 bits wide.  Every target
# is built as   n_j = (h_j << s) | c   where:
#   * c is a FIXED low chunk of s bits, SHARED by all targets in the batch
#     (its popcount pc is high), and
#   * h_j is a per-target high chunk of (60-s) bits with a controlled popcount ph.
#
# Consequences the solver can exploit:
#   * the low s bits agree across the WHOLE batch  ->  the shared sub-exponent c
#     only needs to be assembled ONCE, then reused for every target
#     (meet-in-the-middle split of each exponent into high*2^s + low);
#   * the powers of two (the doubling ladder) are shared by everybody.
# A solver that scores each target with its own square-and-multiply pays for c's
# set bits k times over; the insight pays for them once.
#
# All bit choices are seeded from testId only -> fully deterministic.

# (k, s):  k targets, s shared low bits.  60-s high bits per target.
SPECS = {
    1:  (16, 24),
    2:  (24, 26),
    3:  (32, 28),
    4:  (40, 30),
    5:  (48, 32),
    6:  (60, 34),
    7:  (72, 34),
    8:  (88, 36),
    9:  (100, 36),
    10: (120, 38),
}

W = 60  # every target is exactly W bits wide


def build_chunk(rng, width, popcount, force_top):
    """A `width`-bit value with exactly `popcount` set bits.
    If force_top, bit (width-1) is always set (fixes the value's bit-length)."""
    bits = set()
    if force_top:
        bits.add(width - 1)
    avail = [i for i in range(width) if i not in bits]
    rng.shuffle(avail)
    for i in avail:
        if len(bits) >= popcount:
            break
        bits.add(i)
    v = 0
    for b in bits:
        v |= (1 << b)
    return v


def main():
    tid = int(sys.argv[1])
    k, s = SPECS[tid]
    rng = random.Random(70000 + 7919 * tid)

    hbits = W - s                      # width of the per-target high chunk
    pc = max(2, round(0.5 * s))        # popcount of the shared low chunk c
    ph = max(2, round(0.5 * hbits))    # popcount of each high chunk h_j

    # Shared low chunk c: high popcount so amortization matters.  Top low bit
    # (s-1) forced set only to make c's magnitude stable; it is still shared.
    c = build_chunk(rng, s, pc, force_top=True)

    # Per-target high chunks.  Top bit (hbits-1) forced set for every target so
    # each n_j is exactly 60 bits (stable baseline).  We force target 0 to have
    # an EVEN h (low bit clear) and target 1 an ODD h (low bit set) so that bit
    # s genuinely differs across the batch => the maximal shared suffix is s.
    targets = []
    seen = set()
    idx = 0
    while len(targets) < k:
        h = build_chunk(rng, hbits, ph, force_top=True)
        if idx == 0:
            h &= ~1                     # even  (bit s of n_j clear)
        elif idx == 1:
            h |= 1                      # odd   (bit s of n_j set)
        n = (h << s) | c
        if n in seen:
            continue
        seen.add(n)
        targets.append(n)
        idx += 1

    targets.sort()
    out = [str(k)]
    out.extend(str(t) for t in targets)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
