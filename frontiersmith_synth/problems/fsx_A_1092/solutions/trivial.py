# TIER: trivial
# Reproduces the checker's own baseline: dump ALL cars onto ONE classification
# track (ignore the other Y-1 buckets and the radix structure entirely), then
# for every needed car dig it out one car at a time (scratch-and-restore onto a
# second track), single engine throughout. Always correct, but O(N^2)-ish moves
# and frequent D<->F mode switches -> scores ~0.1.
import sys


def main():
    data = sys.stdin.read().split()
    N, T, L, Y = (int(x) for x in data[:4])
    a, b, s, cap = (int(x) for x in data[4:8])
    rest = data[8:]
    cars = [(int(rest[2 * i]), int(rest[2 * i + 1])) for i in range(N)]

    track0 = list(reversed(cars))
    track1 = []
    track2 = []
    engine_last_mode = {}
    t = 0.0
    lines = []

    def do_move(src_list, dst_list, k, dst_is_bucket, dst_idx):
        nonlocal t
        mode = 'D' if dst_is_bucket else 'F'
        switch = 1 in engine_last_mode and engine_last_mode[1] != mode
        dur = a + b * k + (s if switch else 0)
        engine_last_mode[1] = mode
        src_idx = 0 if src_list is track0 else (1 if src_list is track1 else 2)
        lines.append("1 %.6f %d %d %d" % (t, src_idx, dst_idx, k))
        t += dur
        cut = src_list[-k:]
        del src_list[-k:]
        dst_list.extend(cut)

    while track0:
        do_move(track0, track1, 1, True, 1)

    finals = {tr: [] for tr in range(T)}
    for tr in range(T):
        for slot in range(L - 1, -1, -1):
            target = (tr, slot)
            pos = track1.index(target)
            depth = len(track1) - 1 - pos
            for _ in range(depth):
                do_move(track1, track2, 1, True, 2)
            do_move(track1, finals[tr], 1, False, Y + 1 + tr)
            for _ in range(depth):
                do_move(track2, track1, 1, True, 1)

    print(len(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
