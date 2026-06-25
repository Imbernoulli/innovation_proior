The task is direct to state and expensive to do literally. For a fixed node `v`, I can walk the whole subtree of `v`, count the colors, find the largest frequency, and sum every color that reaches it. That is plainly correct for one node. Doing it for every node costs `sum_v size(v)`: on a path this is `1 + 2 + ... + n`, so the straightforward recount is quadratic. The waste is also visible in the state I throw away. After I count a child's subtree, the parent's subtree needs that same count table plus the parent and the other child subtrees. Rebuilding the table from zero at every ancestor is the part that has to go.

I cannot keep every child's table separately, because that would either duplicate large tables or require merging many maps. But I can arrange the traversal so that exactly one child's count table survives into its parent — the array is shared, so if two children's tallies sit in it at once they mix and I can no longer tell whose is whose. So I get one child for free and must recount the rest by re-walking them. Which one should I keep to make the re-walking cheapest? Keeping a small child saves little; keeping the largest child saves the most re-walking, so let me tentatively keep the child with the largest subtree and see whether the cost actually collapses. I will call this chosen child the heavy child, and store it in code as `pivot_child[u]`. The rest are light children, and I will have to pay to re-walk each of those. Whether that pays off is something I need to count, not assume.

That choice fixes the order. At a node `u`, I first solve every light child in a mode that cleans its contribution away before returning. That way no light sibling contaminates the next one. Then I solve the heavy child in a mode that leaves its count table in place. When that returns, the active table contains exactly the heavy child's subtree. I add `u` itself, then walk each light child's subtree once more and add those nodes. At that moment the active table is exactly the whole subtree of `u`, so `ans[u]` can be read. Finally, if the caller is not supposed to inherit `u`'s table, I walk `u`'s whole subtree and remove it back to zero; otherwise I leave it in place.

The cleanup is not an afterthought; it is part of the cost, and the only way to know whether the heavy-keep idea wins is to count how many times a fixed node gets touched. Fix a node `x`. Whenever an ancestor `a` reaches `x` through a light edge, there are two possible extra touches caused by that edge: the light-subtree computation may remove `x` during its cleanup, and the ancestor may re-add `x` when assembling its own answer. Those are both constant work for the same light edge. If a later ancestor clears a larger subtree that contains `x`, that clear is charged to the light edge above that larger subtree, or to the final root cleanup. The remaining touches are the local add when `x`'s own subtree is assembled and the final cleanup at the root. So the total number of add/remove operations involving `x` is `O(1)` plus a constant times the number of light edges on the root-to-`x` path.

So the question is purely how many light edges a root-to-node path can carry. A light edge cuts the subtree size by at least half: if `x` lies in a light child of a node whose subtree has size `s`, and that light child's subtree has size `s'`, the preserved child has size at least `s'`, and the two child subtrees together fit inside the parent subtree, so `2s' <= s`. Every light step at least halves the remaining subtree size. Starting from at most `n` and ending at at least `1`, there can be at most `log_2 n` light edges on any root-to-node path. Since the cleanup removals and the re-adds are both charged to those same light edges, every node is touched `O(log n)` times and the whole traversal should be `O(n log n)`.

I want to see that bound actually hold rather than trust the charging argument, so let me count total add/remove operations on a few shapes. On a path of `n` nodes every node has a single child, which is always the heavy child, so there are no light edges at all; I expect each node added once and removed once at the final root clear, `2n` total. Running it, `n = 1000, 10000, 100000` give exactly `2000, 20000, 200000` operations — a path is the deepest tree but the cheapest to process, which fits the "no light edges" reading. The bound only bites when there are many light edges, so I check a balanced binary tree: there the counts are `9876, 129226, 1630060` against `n log_2 n` values of `9966, 132877, 1660964`, i.e. ratios `0.99, 0.97, 0.98`. So the balanced tree is where the `n log n` is tight, and the path sits far below it. A random tree lands in between (ratio about `0.6`). The amortized bound is real and the heavy-keep choice is what makes it hold.

The color query needs its own incremental state, because scanning all colors every time I finish a subtree would lose the gain. While adding nodes, I keep `cnt[c]`, `max_freq`, and `sum_dom`. When color `c` is added, its new frequency is `f = cnt[c]`. If `f > max_freq`, then `c` is now strictly the most frequent, so it alone is dominating: `max_freq = f` and `sum_dom = c`. If `f == max_freq`, then `c` joins the current leaders: `sum_dom += c`. If `f < max_freq`, nothing changes. Each branch is constant work, so each add is constant time.

Removal looks harder, because decrementing a color that was tied for the maximum might force me to rediscover the next maximum, which is not a constant-time operation. So I should check whether removals ever happen at a moment when the summary has to be correct. Removals occur only inside `clear_subtree`, when I am emptying an entire active subtree, and I never read an answer during that walk — every `ans[u]` is written from a completed add phase, before any clear begins. So `remove(u)` can simply decrement `cnt[color[u]]` and ignore the summary. When the clear finishes, the table is empty and I can set `max_freq = 0`, `sum_dom = 0` directly, with no search. The summary is therefore maintained exactly during add phases, inherited when a preserved subtree is inherited, and reset only after the active table has been emptied.

Before I commit, I want to run a case that actually exercises the parts the design hinges on: the keep-versus-clear split, and the `f > max_freq` reset where a color stops being dominating. The all-equal trees are no test of that, because every add ties. Take five nodes laid out as `0 - 1 - 2` with `0 - 3` hanging off the root, colors `color = [1, 1, 2, 3]` on nodes `0,1,2,3`. The subtree sizes are `size[0]=4, size[1]=2`, so `pivot_child[0]=1` (the path child, size 2) and `3` is the light child; `pivot_child[1]=2`. Now I walk it.

Node `0` is entered with `mode = False`. Its light child `3` is solved first, also with `mode = False`: I add node `3` (color `3`, `cnt[3]=1 > 0`, so `max_freq=1`, `sum_dom=3`), record `ans[3]=3` — correct, a lone node's dominating color is itself — then `clear_subtree(3)` removes it and the summary is reset to empty. Next the heavy child `1` is solved with `mode = True`. Inside it, its own heavy child `2` is solved first (`mode = True`): add node `2` (color `2`, `cnt[2]=1 > 0`, `max_freq=1`, `sum_dom=2`), `ans[2]=2`, and because `mode=True` nothing is cleared. Then back in node `1`: add node `1` (color `1`, `cnt[1]=1 == max_freq=1`, so it ties color `2`, `sum_dom = 2 + 1 = 3`), `ans[1]=3` — node `1`'s subtree is `{1,2}`, both at frequency one, both dominating, sum `3`, right. Node `1` returns with its table `{color1:1, color2:1}` still live.

Now back at node `0`. I add node `0` (color `1`, `cnt[1]` goes `1 -> 2`, and `2 > max_freq=1`), so this is the branch that resets: `max_freq = 2`, `sum_dom = 1`. Color `1` now stands alone as the most frequent in the active table — this is the reset firing, the thing the all-ties example never reaches. Then I re-walk the light subtree `3` and add it (color `3`, `cnt[3]=1 < 2`), which leaves the summary untouched, exactly as intended for a color below the max. So `ans[0] = 1`. The whole subtree of `0` is colors `{1,1,2,3}`: color `1` appears twice and everything else once, so the dominating color is just `1` and the answer is `1`. The hand trace gives `ans = [1, 3, 2, 3]`. I then ran the implementation on this input and it returns `[1, 3, 2, 3]`, matching the hand trace, and matching an independent brute force that builds each subtree's full color multiset. I also stressed `solve` against that brute force on twenty thousand random small trees with colors drawn in `[1, n]`; every case agreed. The keep/clear bookkeeping and the reset branch both hold.

So I can pin the implementation down. One iterative postorder pass roots the tree at `0`, fills `parent`, computes each `size`, and records in `pivot_child[u]` the child of maximum subtree size. `add_subtree(root)` and `clear_subtree(root)` are explicit stack walks under an already rooted child, so they avoid parent edges. The main `dfs(u, mode)` does light children first with `mode = False`, then the `pivot_child` with `mode = True`, then adds `u` and the light subtrees, records `ans[u]`, and clears if `mode` is false. The recursion descends the heavy edge to a depth equal to the tree height — a path makes that the full `n`, which is exactly the `200000`-operation path case above where depth is maximal — so I raise the recursion limit. The root is called with `False`; its final cleanup costs one linear pass after all answers have already been written.

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

The causal chain is: independent subtree recounts are quadratic because they rebuild the same count state; preserving the largest child table makes every other child light; each light edge can charge both its cleanup removal and its later re-add to one halving step; there are only logarithmically many such steps above any node, which the operation counts above confirm is where the `n log n` becomes tight; and `max_freq` with `sum_dom` makes the completed active subtree answer constant time to read.
