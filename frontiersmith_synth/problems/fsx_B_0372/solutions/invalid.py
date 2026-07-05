# TIER: invalid
# Emits an X for every operation but never inserts any canal exchange: the required
# batches are (almost always) NOT adjacent, so the checker rejects -> score 0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    rows = int(next(it)); cols = int(next(it)); K = int(next(it))
    _ = [(int(next(it)), int(next(it))) for _ in range(K)]
    moves = ["X %d" % t for t in range(K)]
    sys.stdout.write("%d\n%s\n" % (len(moves), "\n".join(moves)))

if __name__ == "__main__":
    main()
