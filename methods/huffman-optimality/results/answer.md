# Huffman coding and its optimality

## Problem

Given a source emitting messages with known probabilities p_1 ≥ p_2 ≥ … ≥ p_m, encode each message as a string of D-ary digits so that the code is **prefix-free** (no codeword is the start of another, so a concatenated stream decodes instantaneously) and the **average length** L = Σ p_i l_i is minimized. The goal is a construction that attains the exact minimum over all prefix codes, with a proof — not merely a code close to the entropy bound.

## Key idea

A prefix code is a D-ary tree with messages at the leaves; l_i is the depth of leaf i. Build the tree **bottom-up**: repeatedly merge the two least-probable nodes (for binary; the D least-probable for a D-ary alphabet) into one combined node of summed probability, recording each merge, until one node remains; then walk the tree assigning a distinct digit per branch. This greedy bottom-up merge is provably optimal — the opposite of the top-down balanced split of Shannon–Fano, which is not.

## The construction (Huffman's algorithm)

1. Put all symbols, with their probabilities, in a min-priority-queue.
2. While more than one node remains: remove the two least-probable nodes, create a parent whose probability is their sum, push it back, and remember the two as its children (assign one a 0-branch, the other a 1-branch).
3. The codeword of a symbol is the sequence of branch digits from the root to its leaf; its length is its depth.

For a D-ary alphabet, merge the D least-probable nodes each step after padding the symbol list with zero-probability dummies until the leaf count is 1 + k(D−1). Equivalently, perform the one smaller initial merge needed to make the remaining count congruent, then merge D at a time.

## Optimality (the exchange-argument proof)

Order p_1 ≥ … ≥ p_m. A code is optimal if Σ p_i l_i is minimal.

**Lemma (an optimal code can be taken in canonical form).** There is an optimal prefix code with:
1. **Anti-ordered lengths.** If p_j > p_k then l_j ≤ l_k. *Proof:* if some l_j > l_k with p_j > p_k, swapping codewords j and k changes L by (p_j − p_k)(l_k − l_j) < 0, contradicting optimality. Hence l_1 ≤ l_2 ≤ … ≤ l_m.
2. **There are sibling codewords at maximum length.** *Proof:* if a maximum-length codeword had no sibling, its parent has one child; deleting its last digit keeps the code prefix-free and strictly lowers L. Contradiction. Therefore a deepest leaf has a deepest sibling, so at least two codewords have maximum length.
3. **The two least-probable symbols can be made deepest siblings.** *Proof:* after anti-ordering, the existence of at least two maximum-length leaves gives l_{m−1} = l_m. Choose any deepest sibling pair. The leaves carrying x_{m−1}, x_m also have maximum length, so exchanging labels among these maximum-depth leaves does not change L. Thus some optimal code has x_{m−1}, x_m at sibling leaves w0 and w1.

**Merge-reduction.** From a canonical code C on m symbols build C′ on m−1 symbols by replacing the sibling pair {x_{m−1}, x_m} (codewords w0, w1 of length l) with a single symbol of probability p_{m−1} + p_m and codeword w (length l−1); other codewords unchanged. Then

  L(C) = Σ_{i≤m−2} p_i l_i + p_{m−1} l + p_m l,
  L(C′) = Σ_{i≤m−2} p_i l_i + (p_{m−1} + p_m)(l−1),
  L(C) − L(C′) = p_{m−1} + p_m,

a constant independent of the code. Since the lemma says some optimum has this sibling form, minimizing the original problem is **identical** to minimizing L(C′) over the merged (m−1)-symbol problem and adding this fixed constant. If C′ were not optimal, a better merged code expanded at the combined leaf would give a better original code.

**Induction.** The reduction lowers the symbol count by one while preserving the optimization exactly. The base case m = 2 is optimal trivially (codewords 0 and 1). Hence the code produced by repeatedly merging the two least-probable nodes is optimal at every size:

> **Theorem.** If C* is the Huffman code and C is any other prefix code for the same source, then L(C*) ≤ L(C).

The same argument extends to D-ary alphabets after the count is padded to 1 + k(D−1): in a full padded tree the D least weights, including any zero dummies, can be placed as deepest siblings, and the offset constant is the sum of those D merged probabilities. Huffman coding is the greedy algorithm that coalesces the least-probable symbols at each stage, and the proof shows this local optimality forces global optimality.

## Where it sits relative to the bounds

By the Kraft inequality (Σ D^{−l_i} ≤ 1 for any prefix code) and the source-coding bound, every prefix code has L ≥ H_D(X), and the Shannon construction l_i = ⌈log_D(1/p_i)⌉ gives H_D(X) ≤ L < H_D(X) + 1. The Huffman code, being the minimum, also satisfies H_D(X) ≤ L(C*) < H_D(X) + 1 and is the smallest L of any prefix code. (Optimality cannot be certified symbol-by-symbol: for (1/3, 1/3, 1/4, 1/12) an optimal code gives a symbol length 3 > ⌈log_2 4⌉ = 2, and optimal length sets need not be unique — so the whole-tree merge argument is essential.)

## Worked checks

For probabilities .25, .25, .2, .15, .15, binary merging gives lengths 2, 2, 2, 3, 3 and average length

    .25·2 + .25·2 + .2·2 + .15·3 + .15·3 = 2.3 bits.

For the same probabilities over a ternary alphabet, merge .15, .15, and .2 into .5, then merge .25, .25, and .5. The lengths are 1, 1, 2, 2, 2, so

    .25·1 + .25·1 + .2·2 + .15·2 + .15·2 = 1.5 ternary digits.

## Heapq implementation

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

p = {'1': .25, '2': .25, '3': .2, '4': .15, '5': .15}
lengths = huffman_code_lengths(p)
avg = sum(p[s] * lengths[s] for s in p)  # 2.3
```
