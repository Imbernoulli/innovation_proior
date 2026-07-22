# TIER: greedy
# Classic Huffman coding on per-chime strike frequency. This optimizes the
# depth-vs-frequency trade-off in isolation but is *blind* to which chimes strike
# back-to-back: it only ever looks at how OFTEN each chime rings, never at WHAT
# follows it, so it is the textbook trap baseline for this family.
import sys
import heapq


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    next(it)  # Dmax, unused by Huffman
    L = int(next(it))
    trace = [int(next(it)) for _ in range(L)]

    count = [0] * (N + 1)
    for s in trace:
        count[s] += 1
    freq = [count[i] + 1 for i in range(N + 1)]  # Laplace smoothing, freq[0] unused

    # heap items: (weight, insertion_order, node_id); node_id < 0 for internal nodes
    heap = []
    counter = 0
    left = {}
    right = {}
    for i in range(1, N + 1):
        heapq.heappush(heap, (freq[i], counter, i))
        counter += 1

    if N == 1:
        codes = {1: "0"}
    else:
        next_internal = -1
        while len(heap) > 1:
            w1, _, n1 = heapq.heappop(heap)
            w2, _, n2 = heapq.heappop(heap)
            merged = next_internal
            next_internal -= 1
            left[merged] = n1
            right[merged] = n2
            heapq.heappush(heap, (w1 + w2, counter, merged))
            counter += 1
        root = heap[0][2]

        codes = {}
        stack = [(root, "")]
        while stack:
            node, pref = stack.pop()
            if node > 0:
                codes[node] = pref if pref else "0"
            else:
                stack.append((left[node], pref + "0"))
                stack.append((right[node], pref + "1"))

    out = [codes[i] for i in range(1, N + 1)]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
