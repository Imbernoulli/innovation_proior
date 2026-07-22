# TIER: strong
# The insight: a run of consecutive rotation days (no recalibration day inside
# it) is just the SAME composed one-period affine map applied over and over --
# an associative operation. Build that composed map once (BUILD) and reach far
# ahead with binary exponentiation (PERIOD n) instead of re-deriving every
# single day. Recalibration days break associativity (they are a genuinely
# different, unbanded map) so they are always handled individually; short
# fringes (less than one period) are cheap enough that per-day banded stepping
# already beats the O(log n) machinery, so the plan compares the two options'
# *declared* costs per run and only pays for BUILD/PERIOD when it wins.
import sys


def parse(tokens):
    it = iter(tokens)

    def nx():
        return int(next(it))

    S = nx(); K = nx(); T = nx(); p = nx(); m = nx()
    for _ in range(S):
        nx()  # x0
    nnz_list = []
    for _k in range(K):
        cnt = 0
        for _i in range(S):
            for _j in range(S):
                if nx() != 0:
                    cnt += 1
        for _i in range(S):
            nx()
        nnz_list.append(cnt)
    pattern = [nx() for _ in range(p)]
    overrides = []
    for _o in range(m):
        t = nx()
        for _i in range(S):
            for _j in range(S):
                nx()
        for _i in range(S):
            nx()
        overrides.append(t)
    return S, K, T, p, m, pattern, overrides, nnz_list


def main():
    data = sys.stdin.read().split()
    S, K, T, p, m, pattern, overrides, nnz_list = parse(data)

    S3 = S ** 3
    period_nnz_sum = sum(nnz_list[pattern[r]] for r in range(p))

    segments = []
    prev = 0
    for t in overrides:
        segments.append((prev, t))
        prev = t + 1
    segments.append((prev, T))

    # Single pass: cover each segment's days (prefix / bulk-period / fringe),
    # then splice in the STEP O for the recalibration day that follows it
    # (every segment except the last is immediately followed by one).
    final_actions = []
    built = False
    for si, (a, b) in enumerate(segments):
        t_cur = a
        if b > a:
            prefix = min((p - (t_cur % p)) % p, b - t_cur)
            for i in range(prefix):
                k = pattern[(t_cur + i) % p]
                final_actions.append(("BAND", k))
            t_cur += prefix

            remaining = b - t_cur
            n_full = remaining // p
            if n_full >= 1:
                build_cost = 0 if built else p * S3
                period_cost = build_cost + 2 * n_full.bit_length() * S3
                direct_cost = n_full * period_nnz_sum
                if period_cost < direct_cost:
                    if not built:
                        final_actions.append(("BUILD",))
                        built = True
                    final_actions.append(("PERIOD", n_full))
                    t_cur += n_full * p
                else:
                    for i in range(n_full * p):
                        k = pattern[(t_cur + i) % p]
                        final_actions.append(("BAND", k))
                    t_cur += n_full * p

            rem2 = b - t_cur
            for i in range(rem2):
                k = pattern[(t_cur + i) % p]
                final_actions.append(("BAND", k))
            t_cur += rem2
        # after this segment, if there is a following override (all but last
        # segment are followed by exactly the override at 'overrides[si]')
        if si < len(overrides):
            final_actions.append(("STEP", "O", si))

    lines = [str(len(final_actions))]
    for a in final_actions:
        lines.append(" ".join(str(x) for x in a))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
