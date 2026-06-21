We need a dynamic set of distinct keys that supports more than insertion: it has to answer order statistics. Given $k$, report the $k$-th smallest key; given a key $v$, report its rank, one plus the number of stored keys strictly less than $v$ — and do both in logarithmic time while keys keep arriving. A binary search tree is the natural home, since the BST property already lays the keys out in sorted order. The fatal flaw of a plain BST is that nothing in ordinary insertion controls the height: insert $1,2,3,\dots$ in order and each new key hangs off the rightmost node, producing a chain of height $n$ where every search, insert, and rank walks the whole chain in $O(n)$. Sorted input is not exotic — it is the common case — so the raw BST is out, and I need balance.

The order-statistics part dictates what every node must carry. To find the $k$-th smallest I stand at a node $t$ and ask how many keys lie in its left subtree; that count plus one is the rank of $t$ within its own subtree. If that rank equals $k$, $t$ is the answer; if $k$ is smaller I descend left; if larger I subtract off the left subtree and $t$ and descend right. Symmetrically, computing the rank of $v$ walks down and, each time the search steps right past a node, banks that node's whole left subtree plus the node itself as keys below $v$. Both queries need exactly one number per node: the size of the subtree rooted there, $s[t] = s[\mathrm{left}[t]] + s[\mathrm{right}[t]] + 1$ with $s[0]=0$ for the null subtree. This field is forced by the queries — it is the thing they read, not optional bookkeeping. The known balancing schemes, by contrast, each demand a *second* field whose only job is to drive rotations: AVL stores a height (or a balance factor) and keeps the two subtrees of every node within one in height; red-black trees store a color bit and four invariants; treaps store a random priority and a heap order; splay trees store nothing but pay amortized cost and surrender worst-case $O(\log n)$ — exactly fatal for a $\mathrm{Select}$ that can hit a transiently tall tree at $O(n)$. So every route leaves me carrying two pieces of per-node state, one of which — the balance field — is dead weight to the queries. The asymmetry is the whole problem: I am forced to keep a size field, and I would rather not also keep a balance field. But size is a more direct measure of lopsidedness than height is; the question that decides everything is whether the size field I already maintain can *be* the balance criterion.

I propose the Size Balanced Tree. It throws away the height, color, and priority entirely and lets $s[t]$ both answer the queries and govern the rotations. The naive size rule — demand that a node's two children never differ in size by more than a constant — fails, because a single leaf insertion bumps one path node's size by one, a one-node perturbation tips a node into violation, and a single rotation relocates a whole subtree and over-corrects. Comparing a node's two children directly is the wrong granularity: a rotation at $t$ does not swap the sizes of $t$'s two children, it relocates grandchildren. So the invariant must be phrased at the grandchild level, so that the very rotation I would fire is the thing that repairs it. Picture a node $T$ with left child $L$ and right child $R$, where $L$ has subtrees $A,B$ and $R$ has subtrees $C,D$; the grandchildren are $A,B,C,D$. The condition I adopt bounds each child against the grandchildren on the other side — a child versus its nephews: don't let $R$'s subtree be smaller than either of $L$'s children, and don't let $L$'s subtree be smaller than either of $R$'s children. For every node $t$,

$$s[\mathrm{right}[t]] \ge s[\mathrm{left}[\mathrm{left}[t]]],\ s[\mathrm{right}[\mathrm{left}[t]]] \qquad\text{(a)}$$
$$s[\mathrm{left}[t]] \ge s[\mathrm{right}[\mathrm{right}[t]]],\ s[\mathrm{left}[\mathrm{right}[t]]] \qquad\text{(b)}$$

In the $T/L/R/A/B/C/D$ picture this is exactly $s[A], s[B] \le s[R]$ and $s[C], s[D] \le s[L]$: each child's subtree is at least as large as each of its nephew subtrees. This is the right granularity because the offending quantity after an insertion is precisely a grandchild that grew, and a single rotation at $T$ promotes a grandchild to a child — the move that can fix a too-large nephew. The invariant talks about the same objects the rotation manipulates.

What makes it work is that this condition alone forces logarithmic height. Let $f[h]$ be the fewest nodes a tree of height $h$ can have while satisfying the condition; if $f[h]$ grows exponentially, then $n \ge f[h]$ forces $h = O(\log n)$. A single node gives $f[0]=1$, and a root plus one child gives $f[1]=2$. For $h>1$, suppose $t$ roots a height-$h$ tree; some child, say $L$, has height $h-1$, so $s[L] \ge f[h-1]$, and inside $L$ there is a child of height $h-2$, a subtree of size at least $f[h-2]$ that is a nephew of $R$. The nephew condition says $R$ is at least as big as each of $L$'s children, so $s[R] \ge f[h-2]$, giving

$$f[h] = s[t] = s[L] + s[R] + 1 \ge f[h-1] + f[h-2] + 1,$$

and the bound is tight — glue a smallest height-$(h-1)$ tree and a smallest height-$(h-2)$ tree under a root — so $f[h] = f[h-1] + f[h-2] + 1$. This Fibonacci-shifted recurrence solves to $f[h] = F_{h+3} - 1$ (with $F_1=F_2=1$): check $(F_{h+2}-1)+(F_{h+1}-1)+1 = F_{h+3}-1$ and $f[0]=F_3-1=1$, $f[1]=F_4-1=2$. By Binet's formula $F_m = (\alpha^m - \beta^m)/\sqrt5$ with $\alpha = (1+\sqrt5)/2$ and $|\beta|<1$, so $F_m$ is within $1/2$ of $\alpha^m/\sqrt5$; inverting $f[h] \le n$ gives worst height $h \le 1.44\log_2(n+1.5) - 1.33 = O(\log n)$ — the same Fibonacci-shaped bound AVL achieves, obtained purely from the size field.

Restoring the condition after an ordinary BST insert is the other half. Insertion is the plain recursive descent: at each node on the path increment $s[t]$ by one, recurse left or right by comparing $v$ to $\mathrm{key}[t]$, and hang the new leaf at the bottom. Only nodes on the path grew, and a grandchild grew by one, so coming back up I run a repair, `_rebalance(t, flag)`, assuming $t$'s subtrees are already fixed and only $t$'s own nephew condition might be off. By the left/right symmetry I only reason through the left-heavy case. If the outer nephew $A = \mathrm{left}[\mathrm{left}[t]]$ is too big, $s[A] > s[R]$, a single right rotation at $T$ lifts $L$ to the top with $A$ still on its left (good — $A$ is big and belongs high) and drops $T$ to be $L$'s right child carrying $B$ over and keeping $R$; afterward $T$ sits over $B,R$ whose nephews are now $R$'s children $C,D$, so I rebalance $T$, and since $L$'s right child changed I rebalance $L$ too. If instead the *inner* nephew $B = \mathrm{right}[\mathrm{left}[t]]$ is the offender, a single rotation won't lift it — $B$ would merely swing to $T$'s left child at the same depth — so this is the double-rotation case: left-rotate $L$ to pull $B$ up above it, then right-rotate $T$ so $B$ rises to the root with $L$ as its left child and $T$ as its right child, the inner mass lifted to the top. The two rotations churn the neighborhood, so I rebalance $L$, then $T$, then $B$ itself. The right-heavy situations are the exact mirrors: outer-right nephew exceeding $s[\mathrm{left}[t]]$ gets a single left-rotate at $T$, inner-right gets a right-rotate of $R$ then a left-rotate of $T$.

A rotation only disturbs the sizes of the two nodes it relinks. Right-rotating at $t$ with $k = \mathrm{left}[t]$, I move $\mathrm{right}[k]$ to become $\mathrm{left}[t]$ and make $t$ the right child of $k$; then $k$ occupies $t$'s old position so $s[k] \leftarrow s[t]$ (set *before* recomputing $t$), and $s[t] \leftarrow s[\mathrm{left}[t]] + s[\mathrm{right}[t]] + 1$. Left rotation is the mirror. Two size assignments, constant time, every other node untouched.

Termination and amortized cost come from a potential: the total depth $SD$, the sum over all nodes of their depths. A short tree has small $SD$, a chain has $SD = \Theta(n^2)$. Every rotation `_rebalance` fires strictly decreases $SD$: in the single-rotation case, right-rotating when $s[A] > s[R]$ lifts the nodes of $A$ up one level and pushes the nodes of $R$ down one, with $B$ unchanged, for a net change $s[R] - s[A] \le -1$; the double case similarly moves the big inner subtree up two levels for a change $\le s[R] - s[B] - 1 < 0$. So no rotating `_rebalance` is gratuitous — each buys a strict drop of at least one in a nonnegative integer potential, which forces termination. Because the height is $O(\log n)$, $SD$ — a sum of $n$ depths each $O(\log n)$ — stays in an $O(n\log n)$ band, and each insert raises $SD$ by only $O(\log n)$. Over $n$ inserts the total rise is $O(n\log n)$ and $SD$ cannot go negative, so the rotating `_rebalance` calls total $O(n\log n)$; the upward passes already make $O(n\log n)$ `_rebalance` calls and each rotating call spawns only a constant number of follow-ups, so total `_rebalance` work is $O(n\log n)$. Divided across those invocations, each `_rebalance` call is $O(1)$ amortized, and each insertion, with $O(\log n)$ calls on its path, is $O(\log n)$ amortized.

One refinement keeps `_rebalance` lean. Coming up from an insertion I know which side I descended into, so I know which property could have broken: a `flag` lets `_rebalance(t, False)` check only the two left-heavy cases and `_rebalance(t, True)` only the two right-heavy cases, halving the checks. After a rotation the four follow-up calls `_rebalance(left[t], False)`, `_rebalance(right[t], True)`, `_rebalance(t, False)`, `_rebalance(t, True)` suffice. The two seemingly missing calls `_rebalance(left[t], True)` and `_rebalance(right[t], False)` are provably no-ops: the violation was created by one insertion so $s[L] \le 2s[R] + 1$, and a passed `_rebalance(t, False)` check gives $s[\mathrm{left}] \le (2s[t]-1)/3$ — a node's lighter side is at most two-thirds of its subtree. Propagating two levels through the double-rotation's inner node $B$ with children $E,F$ gives $s[B] \le (4s[R]+1)/3$ and then $s[E], s[F] \le (8s[R]+3)/9$, whose floor is $\le s[R]$, so the would-be false-side rebalance has nothing to fix; the mirror argument kills the other. Strip the single `_rebalance` line from `_insert` and it is a plain BST — that is how little the balancing costs in code. The two queries read the same size field: `select(k)` uses the in-subtree rank $s[\mathrm{left}[t]] + 1$ to decide return-left-or-right, and `rank(v)` banks $s[\mathrm{left}[t]] + 1$ each time it steps right and adds one at the end to report a rank rather than a strict-less count. I write it array-based, with parallel arrays $\mathrm{key}, \mathrm{left}, \mathrm{right}, s$ and index $0$ the null node.

```python
class OrderStatisticBST:
    """Size Balanced Tree: ordered set with O(log n) insert / select / rank.
    Array-based; index 0 is the null node with s[0] = 0."""

    def __init__(self):
        self.key = [0]; self.left = [0]; self.right = [0]; self.s = [0]
        self.root = 0

    def _new_node(self, v):
        self.key.append(v); self.left.append(0)
        self.right.append(0); self.s.append(1)
        return len(self.key) - 1

    def _left_rotate(self, t):                       # pull right child up
        k = self.right[t]
        self.right[t] = self.left[k]; self.left[k] = t
        self.s[k] = self.s[t]                         # k inherits t's old subtree
        self.s[t] = self.s[self.left[t]] + self.s[self.right[t]] + 1
        return k

    def _right_rotate(self, t):                      # pull left child up
        k = self.left[t]
        self.left[t] = self.right[k]; self.right[k] = t
        self.s[k] = self.s[t]
        self.s[t] = self.s[self.left[t]] + self.s[self.right[t]] + 1
        return k

    def _rebalance(self, t, flag):                    # restore (a)/(b) at t
        L, R, s = self.left, self.right, self.s
        if t == 0:
            return 0
        if not flag:                                 # left side may be too heavy
            if s[L[L[t]]] > s[R[t]]:                  # outer-left nephew -> single
                t = self._right_rotate(t)
            elif s[R[L[t]]] > s[R[t]]:                # inner-left nephew -> double
                L[t] = self._left_rotate(L[t]); t = self._right_rotate(t)
            else:
                return t
        else:                                        # right side (mirror)
            if s[R[R[t]]] > s[L[t]]:                  # outer-right nephew -> single
                t = self._left_rotate(t)
            elif s[L[R[t]]] > s[L[t]]:                # inner-right nephew -> double
                R[t] = self._right_rotate(R[t]); t = self._left_rotate(t)
            else:
                return t
        L[t] = self._rebalance(L[t], False)           # repair the touched nodes
        R[t] = self._rebalance(R[t], True)
        t = self._rebalance(t, False)
        t = self._rebalance(t, True)
        return t

    def _insert(self, t, v):                         # BST insert + one rebalance
        if t == 0:
            return self._new_node(v)
        self.s[t] += 1
        if v < self.key[t]:
            self.left[t] = self._insert(self.left[t], v)
        else:
            self.right[t] = self._insert(self.right[t], v)
        return self._rebalance(t, v >= self.key[t])

    def insert(self, v):
        self.root = self._insert(self.root, v)

    def select(self, k):                             # k-th smallest (1-indexed)
        t = self.root
        while t:
            r = self.s[self.left[t]] + 1             # rank of t in its subtree
            if k == r:   return self.key[t]
            elif k < r:  t = self.left[t]
            else:        k -= r; t = self.right[t]
        raise IndexError("k out of range")

    def rank(self, v):                               # 1 + #keys strictly < v
        t = self.root; ans = 0
        while t:
            if v <= self.key[t]:
                t = self.left[t]
            else:
                ans += self.s[self.left[t]] + 1
                t = self.right[t]
        return ans + 1
```
