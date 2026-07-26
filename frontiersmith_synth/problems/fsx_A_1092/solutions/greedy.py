# TIER: greedy
# The textbook first idea: recognize the yard IS a radix sorter (Y classification
# tracks = digit buckets on key = train_id*L + slot) and run LSD radix passes,
# single car at a time, single engine, exactly the way a stable bucket sort is
# usually taught. This alone already crushes a bucket-blind baseline. But a cut
# move preserves order while a singleton pop-then-push REVERSES it (that's the
# whole reason the bucket-collect step below needs the "process buckets
# Y-1..0" trick) -- so once the working track is fully sorted, peeling cars
# straight off the top one at a time and pushing them onto a train's siding
# would land them in REVERSE slot order. greedy "fixes" this the only way a
# singleton-only mind can: bounce each train's L cars through a scratch track
# first (which un-reverses them) before the final push -- 2 extra single-car
# moves per car instead of the 1 big cut a smarter approach would use.
import sys


def main():
    data = sys.stdin.read().split()
    N, T, L, Y = (int(x) for x in data[:4])
    a, b, s, cap = (int(x) for x in data[4:8])
    rest = data[8:]
    cars = [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(N)]

    D = 1
    maxkey = T * L - 1
    while Y ** D <= maxkey:
        D += 1

    cur = list(reversed(cars))          # track 0, list[-1] = top
    buckets = {b: [] for b in range(1, Y + 1)}

    t = 0.0
    engine_last_mode = {}
    lines = []

    def do_move(src_idx, src_list, dst_idx, dst_list, k, is_bucket_dst):
        nonlocal t
        mode = 'D' if is_bucket_dst else 'F'
        switch = 1 in engine_last_mode and engine_last_mode[1] != mode
        dur = a + b * k + (s if switch else 0)
        engine_last_mode[1] = mode
        lines.append("1 %.6f %d %d %d" % (t, src_idx, dst_idx, k))
        t += dur
        cut = src_list[-k:]
        del src_list[-k:]
        dst_list.extend(cut)

    def key(c):
        return c[0] * L + c[1]

    for r in range(D):
        # distribute: singleton pops from cur into buckets by digit
        while cur:
            c = cur[-1]
            d = (key(c) // (Y ** r)) % Y
            do_move(0, cur, 1 + d, buckets[1 + d], 1, True)
        # collect: buckets Y-1 .. 0, singleton pops back into cur (preserves
        # global ascending-key order after all D passes -- pop-then-push
        # reverses a bucket's build-up order, which is exactly what undoes the
        # reversal the distribute step introduced)
        for bd in range(Y - 1, -1, -1):
            blist = buckets[1 + bd]
            while blist:
                do_move(1 + bd, blist, 0, cur, 1, False)

    # cur is now fully sorted ascending by key: top-to-bottom = train0 slot0,
    # train0 slot1, ..., train(T-1) slot(L-1). A direct singleton peel would
    # push each train's block in REVERSE slot order onto its siding, so bounce
    # every train's L cars through classification track 1 (now empty, reused
    # as scratch) first: that second reversal restores the correct order.
    scratch_idx = 1
    scratch = buckets[1]
    for tr in range(T):
        for _ in range(L):
            do_move(0, cur, scratch_idx, scratch, 1, True)
        for _ in range(L):
            do_move(scratch_idx, scratch, Y + 1 + tr, [], 1, False)

    print(len(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
