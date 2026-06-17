## Problem

Maintain a dynamic set of distinct keys under insertions while supporting the order-statistic queries: report the $k$-th smallest key, and report the rank of a given key (one plus the number of stored keys less than it). The underlying binary search tree must stay balanced -- its height kept $O(\log n)$ -- without expensive extra bookkeeping.

## Code framework

```python
class OrderStatisticBST:
    """Binary search tree storing per-node subtree size, array-based;
    index 0 is the null node with s[0] = 0."""

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

    def _rebalance(self, t, flag):                   # maintenance hook after a BST insert
        # TODO
        return t

    def _insert(self, t, v):                         # BST insert + one maintenance call
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
        # TODO
        pass

    def rank(self, v):                               # 1 + #keys strictly < v
        # TODO
        pass
```
