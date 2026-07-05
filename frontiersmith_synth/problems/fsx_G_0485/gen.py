import sys

# Difficulty ladder: testId 1..10, increasing barcode length / GC content.
# d = 2w - 2 for every case (overlap <= 1 packing) -> even and <= 2w.
TABLE = {
    1:  (15, 4, 6),
    2:  (18, 4, 6),
    3:  (21, 5, 8),
    4:  (24, 5, 8),
    5:  (27, 5, 8),
    6:  (30, 6, 10),
    7:  (33, 6, 10),
    8:  (36, 6, 10),
    9:  (40, 7, 12),
    10: (45, 7, 12),
}

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    n, w, d = TABLE.get(t, TABLE[10])
    sys.stdout.write("%d %d %d\n" % (n, w, d))

if __name__ == "__main__":
    main()
