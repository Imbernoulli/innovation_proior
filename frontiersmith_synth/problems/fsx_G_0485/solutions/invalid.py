# TIER: invalid
# Emits infeasible barcodes: correct length but wrong GC content and duplicates.
import sys

def main():
    p = sys.stdin.read().split()
    n, w, d = int(p[0]), int(p[1]), int(p[2])
    # all-ones (weight n != w) and a duplicate -> guaranteed rejection
    bad = '1' * n
    sys.stdout.write(bad + '\n' + bad + '\n')

if __name__ == "__main__":
    main()
