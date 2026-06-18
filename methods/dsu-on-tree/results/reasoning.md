The task is direct to state and expensive to do literally. For a fixed node `v`, I can walk the whole subtree of `v`, count the colors, find the largest frequency, and sum every color that reaches it. That is plainly correct for one node. Doing it for every node costs `sum_v size(v)`: on a path this is `1 + 2 + ... + n`, so the straightforward recount is quadratic. The waste is also visible in the state I throw away. After I count a child's subtree, the parent's subtree needs that same count table plus the parent and the other child subtrees. Rebuilding the table from zero at every ancestor is the part that has to go.

I cannot keep every child's table separately, because that would either duplicate large tables or require merging many maps. But I can arrange the traversal so that exactly one child's count table survives into its parent — the array is shared, so if two children's tallies sit in it at once they mix and I can no longer tell whose is whose. So I get one child for free and must recount the rest by re-walking them. Which one should I keep? The one whose re-walk I most want to avoid: the child with the largest subtree. I will call this chosen child the heavy child in the reasoning, and store it in code as `pivot_child[u]`. The rest are light children, and I will have to pay to re-walk each of those.

Now the order becomes forced. At a node `u`, I first solve every light child in a mode that cleans its contribution away before returning. That way no light sibling contaminates the next one. Then I solve the heavy child in a mode that leaves its count table in place. When that returns, the active table contains exactly the heavy child's subtree. I add `u` itself, then walk each light child's subtree once more and add those nodes. At that moment the active table is exactly the whole subtree of `u`, so `ans[u]` can be read. Finally, if the caller is not supposed to inherit `u`'s table, I walk `u`'s whole subtree and remove it back to zero; otherwise I leave it in place.

The cleanup is not an afterthought; it is part of the cost. Fix a node `x`. Whenever an ancestor `a` reaches `x` through a light edge, there are two possible extra touches caused by that edge: the light-subtree computation may remove `x` during its cleanup, and the ancestor may re-add `x` when assembling its own answer. Those are both constant work for the same light edge. If a later ancestor clears a larger subtree that contains `x`, that clear is charged to the light edge above that larger subtree, or to the final root cleanup. The remaining touches are the local add when `x`'s own subtree is assembled and the final cleanup at the root. So the total number of add/remove operations involving `x` is `O(1)` plus a constant times the number of light edges on the root-to-`x` path.

A light edge cuts the subtree size by at least half. If `x` lies in a light child of a node whose subtree has size `s`, and that light child's subtree has size `s'`, the preserved child has size at least `s'`. The two child subtrees together fit inside the parent subtree, so `2s' <= s`. Thus every light step at least halves the remaining subtree size. Starting from at most `n` and ending at at least `1`, there can be at most `log_2 n` light edges on any root-to-node path. Since the cleanup removals and the re-adds are both charged to those same light edges, every node is touched `O(log n)` times and the whole traversal is `O(n log n)`.

The color query needs its own incremental state. Scanning all colors every time I finish a subtree would lose the gain. While adding nodes, I keep `cnt[c]`, `max_freq`, and `sum_dom`. When color `c` is added, its new frequency is `f = cnt[c]`. If `f > max_freq`, then `c` is the only color at the new maximum, so `max_freq = f` and `sum_dom = c`. If `f == max_freq`, then `c` joins the current leaders, so `sum_dom += c`. If `f < max_freq`, nothing changes. This makes every add constant time.

Removal looks harder at first, because decrementing a color that was tied for the maximum might require discovering the next maximum. But I never need to answer a query during a removal phase. Removals occur only when I clear an entire active subtree back to empty. During that walk, `remove(u)` only decrements `cnt[color[u]]`; when the walk is over, the correct summary is known without searching: `max_freq = 0` and `sum_dom = 0`. The summary is maintained exactly during add phases, inherited when a preserved subtree is inherited, and reset only after the active table has been cleared.

Before I commit, let me run the smallest case that exercises the keep/clear logic in my head: a root `r` with two single-node children `a` and `b`, three distinct colors. Say `pivot_child[r] = a`. I solve light child `b` first with `mode = False`: its table is just its own color at frequency 1, so `max_freq = 1`, `sum_dom = color[b]`, `ans[b] = color[b]` — a single node's dominating color is its own color, right. Then I clear `b`, and the table and summary go back to empty/zero. Now the heavy child `a` with `mode = True`: same, `ans[a] = color[a]`, and I keep it, so the table still holds `a`. Back at `r`: I add `r` (frequency 1, ties `max_freq = 1`, so `sum_dom += color[r]`), then re-walk light subtree `b` and add it (frequency 1, ties, `sum_dom += color[b]`). All three colors sit at frequency 1, so they all tie as dominators and `sum_dom = color[a] + color[r] + color[b]`. That is exactly the sum of all three colors, which is the right answer for the whole tree. The keep/clear bookkeeping holds.

So I can pin the implementation down. One iterative postorder pass roots the tree at `0`, fills `parent`, computes each `size`, and records in `pivot_child[u]` the child of maximum subtree size. `add_subtree(root)` and `clear_subtree(root)` are explicit stack walks under an already rooted child, so they avoid parent edges. The main `dfs(u, mode)` does light children first with `mode = False`, then the `pivot_child` with `mode = True`, then adds `u` and the light subtrees, records `ans[u]`, and clears if `mode` is false. The recursion descends the heavy edge to a depth equal to the tree height — a path makes that the full `n` — so I raise the recursion limit. The root is called with `False`; its final cleanup costs one linear pass after all answers have already been written.

Here is the final runnable form:

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

The causal chain is: independent subtree recounts are quadratic because they rebuild the same count state; preserving the largest child table makes every other child light; each light edge can charge both its cleanup removal and its later re-add to one halving step; there are only logarithmically many such steps above any node; and `max_freq` with `sum_dom` makes the completed active subtree answer constant time to read.
