We are given a tree of $n$ nodes rooted at node $0$, each carrying a color in $[1, n]$, and for every node $v$ we must report the sum of the dominating colors of its subtree, where a color is dominating if no other color appears more often in that subtree (so all colors tied for the maximum frequency count). The task is trivial to state and expensive to do literally. For a fixed $v$ I can walk the whole subtree, tally the colors, find the largest frequency, and sum every color that reaches it; that is plainly correct for one node. But repeating it independently for every node costs $\sum_v \text{size}(v)$, and on a path that sum is $1 + 2 + \dots + n$, which is $\Theta(n^2)$ — far too slow at $n = 10^5$. The waste is visible in the state I keep discarding: after I tally one child's subtree, the parent needs that very same count table plus the parent itself and the other children, yet the literal approach rebuilds the table from zero at every ancestor. That rebuild from scratch is the part that has to go. I cannot simply keep every child's table separately either, since the count array is shared — if two children's tallies sit in it at once they mix and I can no longer tell whose frequency is whose — so naively preserving all children would force me to duplicate large tables or merge many maps.

The method I propose is DSU on tree (small-to-large on the static count table). The single idea that makes it work is to arrange the traversal so that exactly one child's count table survives into its parent, and to choose that surviving child to be the one whose re-walk I most want to avoid: the child with the largest subtree. I call this the heavy child and store it as `pivot_child[u]`; every other child is light, and I pay to re-walk each light subtree by hand. The order at a node $u$ is then forced. I first recurse into every light child in a mode that cleans its contribution back out before returning, so no light sibling can contaminate the next. Then I recurse into the heavy child in a mode that leaves its count table in place. When that returns, the active table holds exactly the heavy child's subtree; I add $u$ itself, then re-walk each light subtree once and add those nodes, and at that instant the active table is precisely the subtree of $u$, so I read $\text{ans}[u]$. Finally, if my caller is not meant to inherit $u$'s table, I walk $u$'s whole subtree and remove it back to empty; otherwise I leave it standing for the parent to inherit. The cleanup is not an afterthought — it is part of the accounting. Fix a node $x$. Each time an ancestor reaches $x$ across a light edge there are at most two extra touches charged to that edge: the light subtree's cleanup may remove $x$, and the ancestor may re-add $x$ when assembling its own table; both are constant work attributable to that single light edge. A larger later clear that contains $x$ is charged to the light edge above it or to the final root cleanup. The decisive geometric fact is that a light edge at least halves the subtree size: if $x$ sits in a light child of a node whose subtree has size $s$, and that light child's subtree has size $s'$, then the preserved heavy child has size at least $s'$, and the two together fit inside the parent, so $2s' \le s$. Hence at most $\log_2 n$ light edges lie on any root-to-node path, every node is touched $O(\log n)$ times, and the whole traversal runs in $O(n \log n)$ time with $O(n)$ memory.

The color query needs incremental state so that finishing a subtree is constant time rather than a full color scan. While adding nodes I maintain $\text{cnt}[c]$, the active frequency of color $c$, together with $\text{max\_freq}$ and $\text{sum\_dom}$, the sum of the colors currently at that maximum. When I add color $c$ and it reaches frequency $f = \text{cnt}[c]$, the update is

$$
(\text{max\_freq}, \text{sum\_dom}) \leftarrow
\begin{cases}
(f,\; c) & f > \text{max\_freq}\quad(\text{$c$ alone at the new max})\\[2pt]
(\text{max\_freq},\; \text{sum\_dom} + c) & f = \text{max\_freq}\quad(\text{$c$ joins the leaders})\\[2pt]
(\text{max\_freq},\; \text{sum\_dom}) & f < \text{max\_freq}\quad(\text{no change}),
\end{cases}
$$

which is $O(1)$ per add. Removal looks harder, because decrementing a color that was tied for the maximum could in general require searching for the next maximum — but I never query during a removal phase. Removals happen only when I clear an entire active subtree back to empty, so `remove(u)` need only decrement $\text{cnt}[\text{color}[u]]$; once the walk finishes I know the summary without any search, namely $\text{max\_freq} = 0$ and $\text{sum\_dom} = 0$. Thus the summary is maintained exactly during add phases, inherited intact when a preserved subtree is inherited, and reset only after the table has been fully cleared. A small sanity check fixes the bookkeeping: a root $r$ with two single-node children $a, b$ and three distinct colors, $\text{pivot\_child}[r] = a$. Solving light child $b$ first with $\text{mode}=\text{False}$ gives $\text{ans}[b] = \text{color}[b]$ then clears it; the heavy child $a$ with $\text{mode}=\text{True}$ gives $\text{ans}[a] = \text{color}[a]$ and stays; back at $r$ I add $r$ and re-add $b$, all three colors sit at frequency $1$ and tie, so $\text{sum\_dom} = \text{color}[a] + \text{color}[r] + \text{color}[b]$ — exactly the right whole-tree answer. The implementation pins this down with one iterative postorder pass that roots the tree at $0$, fills `parent` and `size`, and records the maximum-size child in `pivot_child[u]`; `add_subtree` and `clear_subtree` are explicit stack walks under an already-rooted child so they never cross a parent edge; and the recursive `dfs(u, mode)` runs light children with $\text{mode}=\text{False}$, the pivot with $\text{mode}=\text{True}$, then adds $u$ and the light subtrees, records the answer, and clears when $\text{mode}$ is false. Because the recursion descends the heavy edge to a depth equal to the tree height — which on a path is the full $n$ — I raise the recursion limit, and the root is invoked with $\text{False}$ so its final linear cleanup runs only after every answer is already written.

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
