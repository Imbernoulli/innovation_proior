# TIER: invalid
# Emits a garbage artifact: every allocation claims an out-of-range berth
# address, and every free carries a nonsense flag. Must score 0 everywhere.
import sys


def main():
    it = sys.stdin.buffer.read().split()
    p = 0
    n = int(it[p + 3]); p += 4
    out = []
    for _ in range(n):
        typ = it[p].decode(); p += 1
        if typ == 'A':
            id_ = int(it[p]); p += 2
            out.append("A %d 999999999" % id_)
        elif typ == 'F':
            id_ = int(it[p]); p += 1
            out.append("F %d 7" % id_)
        else:
            p += 2
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
