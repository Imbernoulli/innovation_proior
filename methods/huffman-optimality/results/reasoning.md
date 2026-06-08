I have a source that emits messages with known probabilities, and I want to replace each message with a string of binary digits so that the average number of digits I transmit is as small as possible. The average is L = Σ p_i l_i, where p_i is the probability of message i and l_i is the length of its code. The catch is that the receiver gets one unbroken stream of digits with no markers between codes and has to chop it back into messages knowing only the codebook. So no code can be the front of another code — if "11" is a code and "111" is also a code, then seeing "11" I can't tell whether that's the whole message or the first two digits of "111". Prefix-free. That's the whole constraint, and it's a real one.

Let me get the geometry straight first, because it's going to do most of the work. A binary code is a binary tree: each edge is a digit, 0 to the left, 1 to the right, and a message lives at a node. The code for a message is the sequence of digits on the path from the root to its node. Prefix-free means no message sits on the path to another message — every message is a *leaf*. So picking a prefix code is exactly picking a binary tree with the messages at the leaves, and l_i is just the depth of leaf i. Now "minimize Σ p_i l_i" reads as "build the tree that puts the heavy leaves shallow and the light leaves deep." Good. The problem is a tree-shaping problem.

How deep can the leaves be? Not arbitrarily — the tree constrains them. Suppose the deepest leaf is at level l_max. A leaf at depth l_i, if I imagine extending the tree all the way down to l_max, would own 2^{l_max − l_i} of the nodes at the bottom level, and because the leaves are prefix-free those owned blocks don't overlap. There are only 2^{l_max} nodes at the bottom, so Σ 2^{l_max − l_i} ≤ 2^{l_max}, i.e. Σ 2^{−l_i} ≤ 1. That's Kraft, and it runs both ways: any lengths with Σ 2^{−l_i} ≤ 1 can be realized as an actual prefix tree (just place the leaves greedily from the shallow end of the unit interval). So I can stop thinking about trees and think about *lengths*: minimize Σ p_i l_i subject to Σ 2^{−l_i} ≤ 1, over positive integers l_i. (Everything here is the same with D symbols instead of 2 and 2 replaced by D; I'll come back to that.)

Where does the floor sit? If I cheat and let the lengths be real numbers, this is a clean Lagrange problem: minimize Σ p_i l_i with Σ 2^{−l_i} ≤ 1. Setting the gradient of Σ p_i l_i + λ Σ 2^{−l_i} to zero gives p_i − λ 2^{−l_i} ln 2 = 0, so 2^{−l_i} ∝ p_i, and pushing the constraint to equality forces 2^{−l_i} = p_i, i.e. l_i = −log_2 p_i. The average there is −Σ p_i log_2 p_i = H, the entropy. And this isn't just the stationary point of a relaxation; it's a genuine lower bound for the integer problem too. Write r_i = 2^{−l_i}/Σ_j 2^{−l_j} and c = Σ 2^{−l_i}. Then L − H = Σ p_i log_2(p_i 2^{l_i}) = Σ p_i log_2(p_i / r_i) − log_2 c = D(p‖r) + log_2(1/c). Relative entropy D(p‖r) ≥ 0, and c ≤ 1 by Kraft so log_2(1/c) ≥ 0. Hence L ≥ H, always, for every prefix code. Nobody beats entropy.

And the ceiling? Round the ideal lengths up: l_i = ⌈log_2(1/p_i)⌉. Then 2^{−l_i} ≤ 2^{−log_2(1/p_i)} = p_i, so Σ 2^{−l_i} ≤ Σ p_i = 1 — Kraft holds, the lengths are realizable — and since log_2(1/p_i) ≤ l_i < log_2(1/p_i) + 1, multiplying by p_i and summing gives H ≤ L < H + 1. So a code within one digit of entropy always exists. Good to know the true minimum is trapped in [H, H+1). But "within a digit" is not "the minimum." I want the *exact* minimum and a proof it's the minimum.

The constructions I have in hand don't give me that. Shannon's own recipe is just l_i = ⌈log_2(1/p_i)⌉ — the ceiling code above. It hits the bound but it's wasteful: it decides each length in isolation from a rounded ideal, and rounding per-symbol throws away digits. The better-known competitor is Fano's. Sort the messages by decreasing probability; cut the sorted list into two consecutive blocks whose total probabilities are as nearly equal as you can manage; give 0 to the top block, 1 to the bottom; then recurse inside each block, cutting again into near-equal halves, until every block is a single message. Each cut adds a digit. It's elegant and it gets to within two digits of entropy. As a tree, it grows from the *root* outward: the very first decision splits *all* the messages into two halves, and that decision fixes the top of the tree before anything about the deep structure is known.

And there's the rub. That top-down balanced cut is a *local* decision made at the place where I know the *least*. "As nearly equal as possible" is a heuristic about probability mass, not about Σ p_i l_i, and the two aren't the same objective. I can cook a distribution where the balanced split forces a longer average code than the genuine best tree — the balanced cut at the root commits me to a shape that a deeper rearrangement could have improved, but top-down I've already paid for it. So Fano's method is provably not optimal in general. It builds the cheap, high-level part of the tree first, on the least information.

Let me stop pushing on the top of the tree, because that's where I'm uncertain, and ask the opposite question: is there any part of the optimal tree I can be *certain* about before I build it? Stare at the objective. Σ p_i l_i. Suppose I had an optimal code and somewhere in it a more probable message had a *longer* code than some less probable message — p_j > p_k but l_j > l_k. Then just swap their two codes: give the shorter code to the heavier message. The change in average length is p_j l_k + p_k l_j − p_j l_j − p_k l_k = (p_j − p_k)(l_k − l_j), and with p_j − p_k > 0 and l_k − l_j < 0 that's strictly negative. I just made an "optimal" code shorter — impossible. So in an optimal code I can always take the lengths to run *opposite* to the probabilities: order p_1 ≥ p_2 ≥ … ≥ p_m and then l_1 ≤ l_2 ≤ … ≤ l_m. The heaviest message is never deeper than a lighter one.

That already tells me something certain about the *bottom* of the tree: the longest code belongs to one of the *least* probable messages. The leaves I'm sure about are the deep ones, not the shallow ones. This is exactly backwards from Fano. He's confident about the root; I'm confident about the leaves. So build from the leaves up.

Push on the deepest leaf. Take one longest code in an optimal tree. Its leaf is at the bottom. Does it have a sibling — another leaf hanging off the same parent? Suppose not: its parent has only this one child. Then that last digit is doing no work — there's no second branch for the decoder to distinguish — so I can delete the last digit of this code, moving the leaf up one level, and the tree is still prefix-free (I haven't created any prefix collision; I've only shortened one codeword that had a lonely parent). That strictly lowers Σ p_i l_i. So an optimal tree can't have a lonely deepest leaf: every maximal-length codeword has a sibling at the same depth. In particular there are at least two codewords of maximum length. Once I arrange the lengths opposite to the probabilities, those two maximum-length positions include x_{m−1} and x_m, the two least probable messages, so l_{m−1} = l_m.

Now choose one deepest sibling pair — same parent, differing only in the final digit, one ending in 0 and one in 1. If the messages sitting there are not literally x_{m−1} and x_m, I can fix that for free. The leaves carrying x_{m−1} and x_m also have maximum length, and exchanging labels among maximum-length leaves does not change any length, so Σ p_i l_i is untouched. After that relabeling, I still have an optimal tree, and in it the two least probable messages are siblings at the bottom, their codes identical except for the last digit.

That's the thing I was sure of, made precise. The two least likely messages can be put at the very bottom as a sibling pair. Their codes are (some common prefix)·0 and (some common prefix)·1.

I don't need to guess the top of the tree at all. There is always an optimal tree in which the two least probable messages are deepest siblings. So I should merge them. Treat the pair {x_{m−1}, x_m} as a single combined message with probability p_{m−1} + p_m, sitting at the shared parent — and now I have a smaller coding problem with m−1 messages. Whatever optimal code I build for the smaller problem, I get the original code back by hanging the two real messages off the combined node, appending a 0 and a 1. I'm building the tree bottom-up: pair off the two lightest, fuse them into one node, and recurse on the lighter list.

I have to check that this merge actually preserves optimality — that solving the smaller problem optimally solves the bigger one optimally, and not just approximately. So let me line up the two codes. Let C be a code on the m messages satisfying everything above (anti-ordered lengths; x_{m−1}, x_m deepest siblings with a common prefix w of length l−1, so their codes are w0 and w1, each of length l). Let C′ be the merged code on m−1 messages: messages 1…m−2 keep their exact codes, and the combined message of probability p_{m−1} + p_m gets the prefix w, of length l−1. The two average lengths are

L(C) = Σ_{i=1}^{m−2} p_i l_i + p_{m−1}·l + p_m·l,

L(C′) = Σ_{i=1}^{m−2} p_i l_i + (p_{m−1} + p_m)·(l−1).

Subtract: L(C) − L(C′) = p_{m−1} l + p_m l − (p_{m−1} + p_m)(l−1) = (p_{m−1} + p_m)·l − (p_{m−1} + p_m)(l−1) = p_{m−1} + p_m. The first sum cancels term for term. So

L(C) = L(C′) + p_{m−1} + p_m,

and the extra piece p_{m−1} + p_m is a *constant* — it doesn't depend on the code at all, only on those two probabilities, which are fixed. That's the whole point. Since every optimum can be rearranged into this sibling form, minimizing L(C) over all original codes is the same as minimizing L(C′) over codes for the merged problem and then adding that fixed constant. If C′ were not optimal for the merged problem, I could replace it by a better merged code and expand the combined leaf back into x_{m−1}, x_m, lowering the original L by the same amount. So the merge doesn't just give a smaller problem; it gives a smaller problem whose optimum maps exactly onto the optimum of the bigger one.

So the recursion is airtight. Minimizing on m messages reduces — exactly, not heuristically — to minimizing on m−1 messages with the two lightest fused. Apply the same reduction to the (m−1)-message problem: sort, fuse the two lightest, recurse. Keep going. The number of messages drops by one each round, and the base case is two messages, where the optimal code is obviously one digit each, 0 and 1 — you can't do better than one digit apiece, and Kraft permits it. Now unwind. At the base, optimal. The merge-reduction says optimality at size k+1 follows from optimality at size k, because the two problems' average lengths differ by a fixed constant — the optimum of one is the optimum of the other lifted by the sum of the two fused probabilities. By induction the code I build by repeatedly fusing the two least probable messages and assigning a 0 and 1 at each fusion is optimal at every size, all the way up to m. There is no prefix code for these probabilities with smaller average length.

Let me say what I actually do, as a procedure. List the messages with their probabilities. Repeat: take the two least probable items currently in the list, remove them, and insert a single combined item whose probability is their sum — and record that these two were paired. When only one item remains, every pairing has been recorded; the pairings form a binary tree, the combined item being the root. Now walk the tree from the root, writing a 0 on one branch and a 1 on the other at every fork; each original message collects the digits on its root-to-leaf path as its code. The length of each message's code is the number of fusions its symbol (or a combined node containing it) took part in — every time it got pulled into a merge, it sank one level deeper.

Trace it to be sure I believe it. Probabilities 0.25, 0.25, 0.2, 0.15, 0.15 for messages 1–5. In binary, the two lightest are 0.15 and 0.15 (messages 4, 5); fuse to 0.30. List: 0.25, 0.25, 0.2, 0.30. Two lightest are 0.2 and 0.25; fuse to 0.45. List: 0.25, 0.30, 0.45. Two lightest 0.25 and 0.30; fuse to 0.55. List: 0.45, 0.55. Fuse to 1.0. Now read out: messages 4 and 5 were merged at the very first step and then carried along inside larger merges twice more, so they're three levels deep — length 3 each, codes 000 and 001. Messages 1, 2, 3 come out at length 2 — codes like 01, 10, 11. Average length = 0.25·2 + 0.25·2 + 0.2·2 + 0.15·3 + 0.15·3 = 0.5 + 0.5 + 0.4 + 0.45 + 0.45 = 2.3 binary digits. The entropy of this source is about 2.286, so I'm 0.014 above the floor and — by the induction — no binary prefix code does better. If I use a ternary alphabet on the same probabilities, the count 5 already has the form 1 + k(3−1). I fuse the three lightest, 0.15, 0.15, and 0.2, into 0.50; then the list is 0.25, 0.25, 0.50, and the final ternary fork uses all three. The two 0.25 messages have length 1, the 0.2 and the two 0.15 messages have length 2, and the average is 0.25·1 + 0.25·1 + 0.2·2 + 0.15·2 + 0.15·2 = 1.5 ternary digits.

One thing I should not have expected, and it sharpens why the global merge is necessary. I cannot certify optimality symbol by symbol against the Shannon length ⌈log_2(1/p_i)⌉. Take probabilities (1/3, 1/3, 1/4, 1/12): the construction gives a code in which one symbol ends up with length 3, which is *longer* than ⌈log_2(1/(1/4))⌉ = 2. So a symbol's optimal length can exceed its own rounded ideal length; the optimal lengths aren't even unique. Optimality is a property of the whole tree, established by the reduction, not something you can read off one leaf at a time. That's exactly why the top-down per-symbol or per-split heuristics miss it and the bottom-up merge catches it.

Now the *D*-ary case, D symbols per fork instead of 2. Each full D-way fusion turns D items into 1, dropping the count by D−1, so the leaf count has to be 1 + k(D−1) if I want every backward step to be a full D-sibling expansion. When the real message count does not have that form, I add zero-probability dummy messages until it does. They can take unused deepest positions and cost nothing. In that padded problem I can use the same exchange picture: a deepest internal parent has D leaf children in a full tree; the D least weights, counting any zero dummies, can be swapped onto those equal-depth leaves; merging those D siblings gives a smaller problem, and the average-length identity becomes L(C) = L(C′) + the sum of the D merged probabilities. If I don't want to describe dummies, the same thing appears as a smaller first merge chosen only to make the remaining count congruent, followed by D-way merges. For D = 2 this congruence is automatic, which is why the binary case never needs padding. With D = 4 and six real messages, 1 + k·3 wants 7, so I add one dummy to reach 7 leaves, fuse the four lightest including the dummy, and proceed. Same induction, same constant-offset merge identity, same conclusion.

So here is the binary core in the standard heap-based form, and the claim that comes with it. Build the tree bottom-up by repeatedly coalescing the two least-probable nodes into one combined node, recording each coalescence; then walk the finished tree assigning one bit per branch. This is a greedy algorithm: at each step it makes the locally forced move, never reconsiders, and yet — because the merge-reduction shows each local fusion preserves the global optimum exactly — the final code is optimal. Local optimality here *is* global optimality, which is the opposite of what happens with the top-down balanced split.

```python
import heapq

def huffman_code_lengths(probs):
    """Return {symbol: codeword length} for an optimal binary prefix code."""
    heap = [(p, i, sym) for i, (sym, p) in enumerate(probs.items())]
    heapq.heapify(heap)
    counter = len(heap)
    # Each loop performs the proven merge: pop the two least weights.
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
```

The chain that makes this the minimum, start to finish: a prefix code is a tree and l_i is leaf depth (so Kraft bounds the achievable lengths and L ≥ H bounds the achievable average); in any optimal tree heavier messages are never deeper, so the longest codes belong to the lightest messages; a deepest leaf must have a sibling or its last digit is wasted, so the two lightest messages can be taken as deepest siblings differing only in their last digit; fusing those two into one node leaves a smaller problem whose average length differs from the original by the fixed constant p_{m−1} + p_m, so minimizing the small problem minimizes the big one *exactly*; by induction from the trivial two-message base, the bottom-up greedy merge of the two (or D) least-probable nodes builds a code of provably minimum average length.
