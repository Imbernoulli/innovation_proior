import sys

# Difficulty ladder: number of gauge stations n grows from small to large.
# The instance is fully described by n; the domain is the fixed unit triangle
# with vertices (0,0), (1,0), (0,1). Anchor instance n = 11.
LADDER = [6, 8, 11, 13, 15, 17, 19, 21, 24, 27]

def main():
    i = int(sys.argv[1])
    if 1 <= i <= len(LADDER):
        n = LADDER[i - 1]
    else:
        # deterministic fallback for out-of-range test ids
        n = 6 + 3 * (i - 1)
        n = max(4, min(48, n))
    sys.stdout.write("%d\n" % n)

if __name__ == "__main__":
    main()
