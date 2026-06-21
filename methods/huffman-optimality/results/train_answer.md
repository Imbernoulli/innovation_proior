We have a source with known message probabilities and want to encode each message as a string of digits so that a concatenated stream can be decoded instantly without separators. That means no codeword can be the prefix of another. The objective is to minimize the average code length L = Σ p_i l_i. The entropy H_D(X) gives a hard lower bound, and Shannon's ceiling construction l_i = ⌈log_D(1/p_i)⌉ gives a code within one digit of it, but it decides each length in isolation and is generally not the true minimum. Shannon–Fano coding improves things by recursively splitting the sorted probabilities into near-equal halves from the root down, but that balanced split is a local decision made with the least information about the final tree; it only guarantees being close to entropy, not attaining the exact minimum. What we need is a construction that provably achieves the smallest possible average length over all prefix codes.

The method I propose is Huffman coding. It builds the code tree bottom-up instead of top-down. Place every symbol with its probability in a min-priority queue. Repeatedly remove the two least-probable nodes, create a parent node whose probability is their sum, and push that parent back into the queue. Continue until one node remains. That sequence of merges defines a binary tree: the original symbols are the leaves, and each merge is an internal node. To read out the codewords, walk from the root to each leaf, writing a 0 on one branch and a 1 on the other; the digits along the path form the codeword, and the path length is the code length.

Why this is optimal follows from an exchange argument and induction. In any optimal code, more probable messages cannot be deeper than less probable ones, because swapping their codewords would strictly lower the average length. Also, some two longest codewords must be siblings: if a deepest leaf had no sibling, its last digit would be doing no work and could be removed, shortening the code. Therefore we can always rearrange an optimal code so that the two least-probable messages are deepest siblings, differing only in their last digit. Merging those two into a single combined message with probability equal to their sum leaves a smaller problem whose average length differs from the original by exactly that sum, a fixed constant independent of the rest of the tree. So minimizing the smaller problem is equivalent to minimizing the original. Repeating this reduction from the full set of messages down to two symbols preserves optimality at every step. Since the two-symbol code is trivially optimal, the code built by greedy bottom-up merging is optimal for any number of messages.

For a D-ary alphabet the same idea applies but each merge combines the D least-probable nodes. To make every reduction a full D-way merge, pad the symbol list with zero-probability dummy messages so the total count has the form 1 + k(D − 1). The dummies fill unused deepest slots and contribute nothing to the average length. The same merge-reduction identity holds with the offset equal to the sum of the D merged probabilities, so the greedy D-way merge again yields the minimum average-length prefix code.

```python
import heapq

def huffman_code_lengths(probs):
    """Return {symbol: codeword length} for an optimal binary prefix code."""
    heap = [(p, i, sym) for i, (sym, p) in enumerate(probs.items())]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, counter, ('merge', n1, n2)))
        counter += 1

    lengths = {}
    def walk(node, depth):
        if isinstance(node, tuple) and node[0] == 'merge':
            walk(node[1], depth + 1)
            walk(node[2], depth + 1)
        else:
            lengths[node] = depth if depth > 0 else 1

    if heap:
        walk(heap[0][2], 0)
    return lengths

def assign_codewords(probs):
    """Return {symbol: codeword} by assigning 0/1 down the merge tree."""
    heap = [(p, i, sym) for i, (sym, p) in enumerate(probs.items())]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, counter, ('merge', n1, n2)))
        counter += 1

    codes = {}
    def walk(node, prefix):
        if isinstance(node, tuple) and node[0] == 'merge':
            walk(node[1], prefix + '0')
            walk(node[2], prefix + '1')
        else:
            codes[node] = prefix or '0'

    if heap:
        walk(heap[0][2], '')
    return codes

def average_length(probs, lengths):
    return sum(probs[s] * lengths[s] for s in probs)

# Example: five symbols with probabilities .25, .25, .2, .15, .15
p = {'1': .25, '2': .25, '3': .2, '4': .15, '5': .15}
lengths = huffman_code_lengths(p)
codes = assign_codewords(p)
avg = average_length(p, lengths)
print({s: (codes[s], lengths[s]) for s in p})
print("Average length:", avg)
```
