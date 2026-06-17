# Size Balanced Tree

## Problem

Maintain a dynamic ordered set of distinct keys under insertion, supporting the
order statistics $\mathrm{Select}(k)$ — the $k$-th smallest key — and
$\mathrm{Rank}(v)$ — one plus the number of keys strictly less than $v$ — while
keeping the binary search tree balanced cheaply.

## Key idea

Every node already stores its subtree size $s[t] = s[\mathrm{left}[t]] +
s[\mathrm{right}[t]] + 1$ (with $s[0]=0$), because that is exactly the field
$\mathrm{Select}$ and $\mathrm{Rank}$ read. The same field becomes the balance
criterion, so no height/color/priority is stored. The **invariant** is that each
child's subtree is at least as large as either of its nephew subtrees: for every
node $t$,

$$s[\mathrm{right}[t]] \ge s[\mathrm{left}[\mathrm{left}[t]]],\ s[\mathrm{right}[\mathrm{left}[t]]]
\qquad\text{(a)}$$
$$s[\mathrm{left}[t]] \ge s[\mathrm{right}[\mathrm{right}[t]]],\ s[\mathrm{left}[\mathrm{right}[t]]]
\qquad\text{(b)}$$

Insertion is the ordinary recursive BST insert (incrementing $s$ down the path),
followed by one call to `_rebalance`, which restores (a)/(b) with rotations:

- **Outer-heavy** nephew (e.g. $s[\mathrm{left}[\mathrm{left}[t]]] > s[\mathrm{right}[t]]$):
  a single rotation (right-rotate $t$).
- **Inner-heavy** nephew (e.g. $s[\mathrm{right}[\mathrm{left}[t]]] > s[\mathrm{right}[t]]$):
  a double rotation (left-rotate $\mathrm{left}[t]$, then right-rotate $t$).

After rotating, `_rebalance` recurses on the touched nodes. Using a `flag` to test only
the side that could have broken (left-heavy when the insert descended left, else
right-heavy), the four follow-up calls
`_rebalance(left[t], False)`,
`_rebalance(right[t], True)`,
`_rebalance(t, False)`, `_rebalance(t, True)` suffice —
the calls `_rebalance(left[t], True)` and
`_rebalance(right[t], False)` are provably no-ops (from
$s[L] \le 2s[R]+1$ one derives $s[E],s[F] \le \lfloor (8s[R]+3)/9 \rfloor \le s[R]$).

## Why it is balanced

Let $f[h]$ be the minimum node count of a size-balanced tree of height $h$. A height-$h$ tree has a
height-$(h-1)$ child (size $\ge f[h-1]$) which itself owns a height-$(h-2)$ subtree
(size $\ge f[h-2]$); by the nephew condition the other child is $\ge f[h-2]$, so

$$f[0]=1,\quad f[1]=2,\quad f[h]=f[h-1]+f[h-2]+1\ (h>1),$$

a Fibonacci-like recurrence with $f[h] = F_{h+3} - 1$. By Binet's formula,
$F_m=(\alpha^m-\beta^m)/\sqrt5$, where $\alpha = (1+\sqrt5)/2$ and
$|\beta|<1$, so $F_m$ is within $1/2$ of $\alpha^m/\sqrt5$. Inverting
$f[h] \le n$ gives worst height
$\le 1.44\log_2(n+1.5) - 1.33 = O(\log n)$ — the same bound as AVL, from the size
field alone.

**`_rebalance` is $O(1)$ amortized per invocation.** Let $SD$ be the sum
of all node depths. Every rotation `_rebalance` fires strictly decreases $SD$ (single
case: by $s[\mathrm{left}[\mathrm{left}[t]]] - s[\mathrm{right}[t]] \ge 1$; double
case: by $> 1$). Since the height is $O(\log n)$, $SD$ stays in an
$O(n\log n)$ band, and each insert raises $SD$ by only $O(\log n)$; over $n$
inserts the rotating `_rebalance` calls total $O(n\log n)$. The upward insertion paths
also make $O(n\log n)$ `_rebalance` calls, and each rotating call creates only a
constant number of follow-up calls, so total `_rebalance` work is $O(n\log n)$. Thus
each `_rebalance` invocation is $O(1)$ amortized, insertion is $O(\log n)$ amortized,
and Select/Rank are $O(\log n)$ worst case from the height bound.

## Code

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

## Complexity

- **Height:** $\le 1.44\log_2(n+1.5) - 1.33 = O(\log n)$ (Fibonacci bound).
- **`_rebalance`:** $O(1)$ amortized per invocation ($O(n\log n)$ total
  work over the $O(n\log n)$ invocations made by $n$ inserts).
- **Insert:** $O(\log n)$ amortized.
- **Select / Rank:** $O(\log n)$ worst case.
- **Per-node state:** key, two child pointers, and the subtree size $s$ — and $s$
  is the field the order-statistics queries consume, so there is no balance field
  that the queries cannot use.
