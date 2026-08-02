# TIER: greedy
"""The obvious first implementation: "backfill-then-cutover". Every backfill
tick blindly overwrites the NEW store with whatever the scanner read (no
version check -- the natural first-pass port of a single-threaded copy
script), and reads get cut over to NEW as soon as the backfill phase looks
done, i.e. right after the LAST backfill tick in the whole timeline. This is
clean whenever nothing else was concurrently writing the same keys, but a
stale backfill batch that physically lands *after* a newer live write to the
same key silently clobbers it -- and the read-checks in this problem's
concurrent-load cases are placed exactly where that clobber becomes visible.
"""
import sys


def main():
    head = sys.stdin.readline().split()
    K, T, M = int(head[0]), int(head[1]), int(head[2])
    sys.stdin.readline()  # baseline values, unused

    last_b_idx = -1
    for i in range(T):
        parts = sys.stdin.readline().split()
        if parts[0] == 'B':
            last_b_idx = i

    C = last_b_idx + 1  # cut over right after the backfill phase ends
    flags = ["0"] * M   # unconditional overwrite -- no version discipline

    sys.stdout.write(f"{C}\n{' '.join(flags)}\n")


if __name__ == "__main__":
    main()
