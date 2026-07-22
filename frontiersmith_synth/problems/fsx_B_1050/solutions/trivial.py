# TIER: trivial
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    next(it); next(it)  # ALPHA, LAMBDA -- unused
    for _ in range(N):
        next(it); next(it)  # points -- unused
    for _ in range(N * N):
        next(it)  # bonus matrix -- unused

    # Naive fixed index-pairing: merge (1,2), (3,4), ... requeue, repeat.
    # Ignores both geometry and the bonus matrix -- this is exactly the
    # checker's own internal baseline construction.
    queue = list(range(1, N + 1))
    merges = []
    next_id = N + 1
    while len(queue) > K:
        a = queue.pop(0)
        b = queue.pop(0)
        merges.append((a, b))
        queue.append(next_id)
        next_id += 1

    out = [str(len(merges))]
    for a, b in merges:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
