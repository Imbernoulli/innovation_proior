# TIER: strong
"""The insight: protect every backfill write with a version/watermark check
(only overwrite the NEW store if the backfill's snapshot version is strictly
newer than what's already there), which makes every backfill tick provably
non-destructive -- a stale batch landing after a fresher live write simply
no-ops instead of clobbering it. That alone already fixes every lost-update
case greedy fails on.

The second half of the insight ("cutover-verification"): once every backfill
write is watermark-safe, a key's NEW-store value is guaranteed correct from
the tick of its *last* live write onward (or from its *first* backfill touch
onward, if it is never live-written again) -- earlier writes to the same key
can never un-sync it once that tick has passed. So the earliest SAFE cutover
is not "wait for the whole backfill phase", it is: for every read-check tick,
only the read-checks aimed at a key that is not yet synced by that tick
actually constrain the cutover; take the tightest one. Reads on keys that are
already synced impose no constraint at all, letting the cutover move much
earlier than greedy's blanket "wait for backfill to finish".
"""
import sys


def main():
    head = sys.stdin.readline().split()
    K, T, M = int(head[0]), int(head[1]), int(head[2])
    sys.stdin.readline()  # baseline values, unused

    ops = []
    for i in range(T):
        parts = sys.stdin.readline().split()
        if parts[0] == 'R':
            ops.append(('R', int(parts[1])))
        else:
            ops.append((parts[0], int(parts[1])))

    last_live = {}   # key -> last tick index with an 'L' op
    first_touch = {}  # key -> first tick index touching the key at all (L or B)
    for i, op in enumerate(ops):
        if op[0] in ('L', 'B'):
            k = op[1]
            if k not in first_touch:
                first_touch[k] = i
            if op[0] == 'L':
                last_live[k] = i

    def sync_time(k):
        if k in last_live:
            return last_live[k]
        if k in first_touch:
            return first_touch[k]
        return None  # never touched -- can never be safely read post-cutover

    min_safe_C = 0
    for i, op in enumerate(ops):
        if op[0] == 'R':
            k = op[1]
            st = sync_time(k)
            if st is None or st > i:
                min_safe_C = max(min_safe_C, i + 1)

    C = min(min_safe_C, T)
    flags = ["1"] * M  # blanket version/watermark discipline: always conditional

    sys.stdout.write(f"{C}\n{' '.join(flags)}\n")


if __name__ == "__main__":
    main()
