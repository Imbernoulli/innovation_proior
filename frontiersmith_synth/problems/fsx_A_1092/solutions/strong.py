# TIER: strong
# Same LSD radix skeleton as greedy (Y buckets = digit buckets on key =
# train_id*L + slot; a bucket's build-up order MUST be undone one car at a
# time -- a cut move preserves order, a singleton pop-then-push reverses it,
# and only the reversal restores the correct global order after distributing,
# so the distribute/collect passes below stay exactly as fine-grained as
# greedy's). The real insight is recognizing WHERE a cut move's order
# preservation is actually an asset instead of a liability: once the working
# track is fully sorted, each train's L cars already sit contiguously in
# EXACTLY the order its siding needs (slot 0 nearest the top) -- so one single
# cut of size L straight onto an empty siding reproduces what greedy's
# scratch-and-bounce needed 2L singleton moves to achieve. Combine that with
# engine choreography: engine 1 executes every distribute-type move and NEVER
# anything else; engine 2 executes every collect-type move (bucket -> working
# track, or working track -> siding) and NEVER anything else. Neither engine's
# move type ever changes, so the s-tick mode-switch penalty is never paid
# (the lead is still a strict mutex -- only one move executes at a time -- but
# neither engine ever has to re-line the switches for the other's kind of
# move).
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

    cur = list(reversed(cars))
    buckets = {b: [] for b in range(1, Y + 1)}

    t = 0.0
    engine_last_mode = {}
    lines = []

    def do_move(engine, src_idx, src_list, dst_idx, dst_list, k, is_bucket_dst):
        nonlocal t
        mode = 'D' if is_bucket_dst else 'F'
        switch = engine in engine_last_mode and engine_last_mode[engine] != mode
        dur = a + b * k + (s if switch else 0)
        engine_last_mode[engine] = mode
        lines.append("%d %.6f %d %d %d" % (engine, t, src_idx, dst_idx, k))
        t += dur
        cut = src_list[-k:]
        del src_list[-k:]
        dst_list.extend(cut)

    def key(c):
        return c[0] * L + c[1]

    for r in range(D):
        # engine 1: distribute, singleton (destination depends on each car's digit)
        while cur:
            c = cur[-1]
            d = (key(c) // (Y ** r)) % Y
            do_move(1, 0, cur, 1 + d, buckets[1 + d], 1, True)
        # engine 2: collect, singleton pops per bucket, processed Y-1 .. 0
        # (this is the reversal that undoes the distribute step's reversal --
        # batching a whole bucket into one cut would PRESERVE its build-up
        # order instead and silently corrupt the sort, so this stays singleton)
        for bd in range(Y - 1, -1, -1):
            blist = buckets[1 + bd]
            while blist:
                do_move(2, 1 + bd, blist, 0, cur, 1, False)

    # cur is fully sorted ascending by key: top-to-bottom = train0's L cars
    # (slot 0..L-1), then train1's L cars, etc. A single cut PRESERVES that
    # order, so engine 2 peels one contiguous L-sized block per train straight
    # onto an empty siding and it lands already in slot order.
    while cur:
        tr = cur[-1][0]
        do_move(2, 0, cur, Y + 1 + tr, [], L, False)

    print(len(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
