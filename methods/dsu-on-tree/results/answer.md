# DSU on tree for dominating colors

## Method

For each node `u`, keep the color counts for one child subtree alive and rebuild
only the other child subtrees around it. The child whose subtree is largest is
stored as `pivot_child[u]`; all other children are processed, answered, and
cleared before the preserved child is processed. Then `u` and the cleared child
subtrees are added back, so the active count table becomes exactly the subtree
of `u`.

The query state is maintained during additions:

- `cnt[c]`: current active frequency of color `c`
- `max_freq`: maximum active frequency
- `sum_dom`: sum of colors whose active frequency equals `max_freq`

When a color `c` is added and reaches frequency `f`, reset the answer state if
`f > max_freq`, append `c` if `f == max_freq`, and otherwise leave it unchanged.
Removal is only used while clearing a whole active subtree, and no answer is read
mid-clear, so removal only decrements `cnt`; after the clear finishes,
`max_freq` and `sum_dom` are reset to zero.

## Complexity

The amortized bound counts cleanup as well as re-addition. For a node `x`, every
light edge above it can cause a constant number of extra touches: a cleanup
removal from the light subtree and a later re-add when the ancestor assembles its
own active table. A light edge at least halves subtree size, so there are at most
`log_2 n` light edges on any root-to-node path. Including the local add and the
final root cleanup, every node is touched `O(log n)` times. The total running
time is `O(n log n)`, and the memory use is `O(n)`.

## Code

```python
import sys
from sys import setrecursionlimit


def solve(n, color, edges):
    """color[v] is the color of node v (0-based); edges is a list of (a, b)
    undirected tree edges. Return ans where ans[v] is the sum of all
    dominating colors in the subtree of v (tree rooted at node 0)."""
    g = [[] for _ in range(n)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)

    size = [1] * n
    pivot_child = [-1] * n
    parent = [-1] * n
    stack = [(0, -1, False)]
    while stack:
        u, p, processed = stack.pop()
        if processed:
            best = 0
            for w in g[u]:
                if w != p:
                    size[u] += size[w]
                    if size[w] > best:
                        best = size[w]
                        pivot_child[u] = w
            continue
        parent[u] = p
        stack.append((u, p, True))
        for w in g[u]:
            if w != p:
                stack.append((w, u, False))

    cnt = [0] * (n + 1)
    max_freq = 0
    sum_dom = 0
    ans = [0] * n

    def add(u):
        nonlocal max_freq, sum_dom
        c = color[u]
        cnt[c] += 1
        f = cnt[c]
        if f > max_freq:
            max_freq = f
            sum_dom = c
        elif f == max_freq:
            sum_dom += c

    def remove(u):
        cnt[color[u]] -= 1

    def add_subtree(root):
        st = [root]
        while st:
            u = st.pop()
            add(u)
            for w in g[u]:
                if w != parent[u]:
                    st.append(w)

    def clear_subtree(root):
        st = [root]
        while st:
            u = st.pop()
            remove(u)
            for w in g[u]:
                if w != parent[u]:
                    st.append(w)

    setrecursionlimit(1 << 20)

    def dfs(u, mode):
        nonlocal max_freq, sum_dom
        for w in g[u]:
            if w != parent[u] and w != pivot_child[u]:
                dfs(w, False)
        if pivot_child[u] != -1:
            dfs(pivot_child[u], True)
        add(u)
        for w in g[u]:
            if w != parent[u] and w != pivot_child[u]:
                add_subtree(w)
        ans[u] = sum_dom
        if not mode:
            clear_subtree(u)
            max_freq = 0
            sum_dom = 0

    dfs(0, False)
    return ans


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    color = [int(next(it)) for _ in range(n)]
    edges = []
    for _ in range(n - 1):
        x = int(next(it)) - 1
        y = int(next(it)) - 1
        edges.append((x, y))
    ans = solve(n, color, edges)
    sys.stdout.write(" ".join(map(str, ans)) + "\n")


if __name__ == "__main__":
    main()
```
