import sys, random


def make_chain(k, y0, x_start, value_base, value_step, span=1):
    """A chain of k parts on row y0.  Part idx's footprint sits at
    x = x_start + 3*idx (a single cell).  Part idx's corridor is a strip
    immediately to its right that reaches into the footprints of the next
    `span` chain parts (idx+1 .. idx+span), so footprint(idx+j) intersects
    corridor(idx) for j=1..span, but the reverse never holds.  Consequence:
    for idx and all of idx+1..idx+span to jointly survive, idx must be
    attempted (and free) *before any of them is installed* -- the unique
    fully-safe order is ascending idx (0,1,...,k-1).  Installing a higher
    idx before a lower one it dominates seals that lower idx's corridor for
    good, and with span>1 a single early insertion can foreclose several
    not-yet-installed parts' corridors at once.  value = value_base +
    value_step*idx, so value grows (or shrinks, if value_step<0) linearly
    with position in the safe order.
    """
    parts = []
    for idx in range(k):
        x0 = x_start + 3 * idx
        footprint = (x0, y0, 1, 1)
        reach = min(span, k - 1 - idx)
        cw = max(1, 3 * reach)
        corridor = (x0 + 1, y0, cw, 1)
        value = value_base + value_step * idx
        parts.append({"value": value, "f": footprint, "c": corridor})
    return parts


def make_neutral(y, x_start, value):
    footprint = (x_start, y, 1, 1)
    corridor = (x_start + 1, y, 2, 1)  # points into open, unused space
    return {"value": value, "f": footprint, "c": corridor}


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20000 + testId)

    k_main = min(4 + (testId - 1) // 2, 8)
    m_pairs = 4
    neutral_count = 2

    W = 3 * k_main + 10
    H = 1 + m_pairs + neutral_count

    # -- main chain: value GROWS with required position (idx k-1 is the
    #    most valuable and must go LAST).  Each corridor reaches into the
    #    next TWO chain footprints (span=2), so one careless early
    #    insertion can seal off two not-yet-installed parts' access at
    #    once.  File order lists it in descending-idx order, i.e. exactly
    #    the order a value-first pass would also choose -- so an
    #    unstructured index-order pass and a plain highest-value-first pass
    #    make the identical mistake here. --
    chain = make_chain(k_main, y0=0, x_start=0,
                        value_base=15, value_step=20, span=2)
    ordered_blocks = [list(reversed(chain))]

    # -- access-fragile high-value pairs: value SHRINKS with required
    #    position (idx0 is the expensive, must-go-first part; idx1 is a
    #    cheap part that seals idx0's corridor if it jumps the queue).
    #    File order lists the cheap part before the expensive one, so a
    #    plain index-order pass always seals the expensive part's corridor
    #    -- while sorting by value first (installing the expensive part
    #    immediately) always saves it. --
    for j in range(m_pairs):
        hi = rng.randint(30, 42)
        lo = rng.randint(5, 8)
        pair = make_chain(2, y0=1 + j, x_start=0, value_base=hi, value_step=lo - hi, span=1)
        ordered_blocks.append(list(reversed(pair)))

    # -- a couple of isolated, non-interacting parts for baseline mass --
    neutrals = []
    for t in range(neutral_count):
        y = 1 + m_pairs + t
        val = rng.randint(8, 14)
        neutrals.append(make_neutral(y, 0, val))
    ordered_blocks.append(neutrals)

    all_parts = [p for block in ordered_blocks for p in block]
    n = len(all_parts)

    lines = ["%d %d %d" % (n, W, H)]
    for p in all_parts:
        fx, fy, fw, fh = p["f"]
        cx, cy, cw, ch = p["c"]
        lines.append("%d %d %d %d %d %d %d %d %d" %
                      (p["value"], fx, fy, fw, fh, cx, cy, cw, ch))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
