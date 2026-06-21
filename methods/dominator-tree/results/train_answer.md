I have a directed flowgraph $G$ with a start vertex $s$, and for every vertex $v$ reachable from $s$ I want its immediate dominator: the closest vertex other than $v$ that lies on every path from $s$ to $v$. A vertex $u$ dominates $v$ exactly when deleting $u$ disconnects $v$ from $s$, so $s$ dominates everything, $v$ dominates itself, and the dominators of a fixed $v$ are totally ordered by "is dominated by" — on any $s\to v$ path they all appear, and whichever appears first dominates the rest. They form a chain $s=u_0,u_1,\dots,u_k=v$, and $\mathrm{idom}(v)=u_{k-1}$, the last bottleneck before $v$. Wiring $\mathrm{idom}(v)\to v$ for every reachable $v$ produces the dominator tree rooted at $s$. The entire task is to find that one parent pointer per vertex, and the graph may have cycles, self-loops, parallel edges, and unreachable vertices, with $n,m$ up to $10^5$ or more.

The literal reading of the definition gives the first algorithm: delete each vertex $u$ in turn, search from $s$, and every vertex that becomes unreachable has $u$ as a dominator. That correctly recovers the full dominator set of every vertex, from which the immediate one — the dominator dominated by all the others, equivalently the deepest bottleneck — is read off. But it costs one $O(n+m)$ search per deleted vertex, so $O(nm)$, around $10^{10}$ at scale, and it materializes the entire $n\times n$ dominator-set table only to extract a single parent per vertex. The fixpoint refinement keeps the intersection idea — a non-self dominator of $v$ must dominate every predecessor of $v$, since every $s\to v$ path enters through one, so $\mathrm{Dom}(v)=\{v\}\cup\bigcap_{u\in\mathrm{pred}(v)}\mathrm{Dom}(u)$ — and iterates with bitsets in reverse postorder to roughly $O(n^2)$. Still quadratic, and still carrying full $n$-bit sets when I want only one number. The waste is the same in spirit: I am computing sets to intersect when I should compute the single immediate dominator directly, in near-linear time, never building a set.

The solution is the Lengauer-Tarjan algorithm, and its engine is a quantity called the semidominator. The right scaffold is a depth-first search from $s$, which produces a DFS tree $T$ with parent $\mathrm{fa}(v)$ and a preorder number $\mathrm{dfn}$; I write $u<v$ for $\mathrm{dfn}(u)<\mathrm{dfn}(v)$. The tree path $s\rightsquigarrow v$ in $T$ is a genuine $s\to v$ path, so $\mathrm{idom}(v)$ — being on every such path — must be a $T$-ancestor of $v$. That alone confines $\mathrm{idom}(v)$ to the vertical strip from $s$ down to $v$. It is not simply $\mathrm{fa}(v)$, because a forward or cross edge can enter $v$'s subtree from outside, dodging a tree-path ancestor; the question is precisely which ancestors get bypassed by non-tree edges and which survive as true bottlenecks. A path can reach $v$ while skipping ancestors only by leaving the tree, wandering through high-numbered vertices (those DFS reached after $v$, sitting to the right in preorder), and re-entering near $v$. So I define, for each $v$, the smallest-$\mathrm{dfn}$ vertex $u$ such that some path $u=x_0\to x_1\to\dots\to x_k=v$ has every interior vertex $x_1,\dots,x_{k-1}$ numbered $>v$, and call it $\mathrm{sdom}(v)$. The interior-all-larger condition is the formal version of "the detour touches only stuff DFS reached after $v$," and that is exactly how a high ancestor $u$ can reach $v$ without being forced through the in-between ancestors, since any ancestor strictly between $u$ and $v$ is numbered $<v$, not $>v$. The tree edge $\mathrm{fa}(v)\to v$ has no interior, so $\mathrm{fa}(v)$ is always a candidate and $\mathrm{sdom}(v)<v$; a short DFS-preorder argument shows $\mathrm{sdom}(v)$ is in fact a proper $T$-ancestor of $v$. And because the defining detour runs from $\mathrm{sdom}(v)$ through vertices that are not strict ancestors of $v$ — none of which dominates $v$ — the true bottleneck $\mathrm{idom}(v)$ cannot be one of those interior vertices, so $\mathrm{idom}(v)$ is an ancestor of $\mathrm{sdom}(v)$. The immediate dominator is now squeezed onto the tree path between $s$ and $\mathrm{sdom}(v)$.

What makes this tractable is that $\mathrm{sdom}$ can be computed without enumerating detour paths. Unfolding a minimizing path and looking at its last edge $w\to v$ splits the candidates into two families: predecessors $w$ of $v$ with $w<v$, each contributing itself; and, for every predecessor $w>v$ and every $T$-ancestor $a$ of $w$ with $a>v$, the value $\mathrm{sdom}(a)$ — because a detour realizing $\mathrm{sdom}(a)$ (interior all $>a>v$) can be extended down the tree from $a$ to $w$ (vertices all $>v$) and then across $w\to v$, giving a valid detour for $v$. A careful argument using the DFS ancestor-sandwiching property (on such a stretch, a vertex with $v_i<v_j$ but $v_i>v$ must be an ancestor of $v_j$) shows the minimizing path's relevant prefix is itself a valid semidominator path for the right intermediate vertex, so the minimum over the two families equals $\mathrm{sdom}(v)$ exactly:
$$\mathrm{sdom}(v)=\min\Big(\{\,w:(w,v)\in E,\ w<v\,\}\ \cup\ \{\,\mathrm{sdom}(a):a>v,\ a\text{ a }T\text{-ancestor of a predecessor }w>v\text{ of }v\,\}\Big),$$
minimizing by $\mathrm{dfn}$. The path quantifier is gone — I only look at $v$'s predecessors and, for each, the minimum $\mathrm{sdom}$ over its ancestors numbered $>v$.

The "over all ancestors with $a>v$" is dispatched by processing vertices in decreasing $\mathrm{dfn}$ order. When I am about to compute $\mathrm{sdom}(v)$, every vertex numbered $>v$ has its final $\mathrm{sdom}$, and the ancestors $a>v$ of a predecessor $w$ are exactly the already-processed vertices on $w$'s tree path above the current frontier. So I maintain a forest of processed vertices, each linked to its tree parent once processed, and define $\mathrm{eval}(w)$ as the query that walks the processed portion of $w$'s tree path and returns the vertex of minimum $\mathrm{sdom}$ encountered — returning $w$ itself when no link above $w$ is installed yet. This single rule unifies both families: an unprocessed smaller predecessor contributes $w$ (its $\mathrm{sdom}$ is still seeded to itself), a processed root contributes its computed $\mathrm{sdom}(w)$, so
$$\mathrm{sdom}(v)=\min_{w\in\mathrm{pred}(v)}\mathrm{sdom}(\mathrm{eval}(w)).$$
A naive walk per $\mathrm{eval}$ would be $O(nm)$ again, but this is precisely the workload a path-compressing disjoint-set structure handles: when $\mathrm{eval}(w)$ walks to the root I splice every vertex on the walk directly to the root and carry along a $\mathrm{label}$ recording the minimum-$\mathrm{sdom}$ witness on the compressed segment above it, comparing by $\mathrm{dfn}(\mathrm{sdom}(\cdot))$. Subsequent walks are short because the tree flattens, and the total cost across all $\mathrm{eval}$s is near-linear — $O(m\log n)$ with plain compression as used here, $O(m\,\alpha(m,n))$ with balanced linking.

Knowing $\mathrm{sdom}$ pins $\mathrm{idom}(v)$ between $\mathrm{sdom}(v)$ and $s$; what decides where is the stretch of vertices strictly between $\mathrm{sdom}(v)$ and $v$. The detour defining $\mathrm{sdom}(v)$ starts at $\mathrm{sdom}(v)$, so it does not bypass $\mathrm{sdom}(v)$ itself, but a vertex on that stretch may reach even higher and so bypass $\mathrm{sdom}(v)$. Let $u=\mathrm{eval}(v)$ be the vertex of minimum semidominator on that stretch — conveniently the same query right after $v$ is linked. If $\mathrm{sdom}(u)=\mathrm{sdom}(v)$, nothing reaches above $\mathrm{sdom}(v)$, so $\mathrm{sdom}(v)$ genuinely dominates $v$ and, being lowest, equals $\mathrm{idom}(v)$. If $\mathrm{sdom}(u)<\mathrm{sdom}(v)$, some $u$ bypasses $\mathrm{sdom}(v)$, so $\mathrm{idom}(v)$ is strictly higher, and the gift is that it coincides with $\mathrm{idom}(u)$ — whatever bottleneck finally controls $u$ controls $v$, since $v$ hangs below $u$. Thus
$$\mathrm{idom}(v)=\begin{cases}\mathrm{sdom}(v) & \text{if }\mathrm{sdom}(u)=\mathrm{sdom}(v),\\ \mathrm{idom}(u) & \text{otherwise,}\end{cases}\qquad u=\mathrm{eval}(v).$$
The second case references $\mathrm{idom}(u)$, which may be unknown when I reach $v$, so I defer it: drop $v$ into $\mathrm{bucket}(\mathrm{sdom}(v))$, and when I link $v$ under $\mathrm{fa}(v)$, drain $\mathrm{bucket}(\mathrm{fa}(v))$ — for each $x$ there, $\mathrm{fa}(v)=\mathrm{sdom}(x)$, the stretch from $\mathrm{sdom}(x)$ down is now linked, so I run $u=\mathrm{eval}(x)$ and apply the rule, recording the witness $u$ tentatively in the second case. Draining at the parent rather than at $v$ is what makes the linked region exactly the stretch the $\mathrm{eval}$ must see. A final pass in increasing $\mathrm{dfn}$ order resolves the deferrals: for each $v$ whose recorded $\mathrm{idom}(v)$ is not already its semidominator, set $\mathrm{idom}(v)\leftarrow\mathrm{idom}(\mathrm{idom}(v))$; increasing order guarantees the witness was finalized first. Then $\mathrm{idom}(s)=s$ and the tree is complete. Unreachable vertices keep $\mathrm{dfn}=0$, never enter the sweeps, and are skipped as predecessors; self-loops and parallel edges only repeat harmless candidates, so the raw graph goes in unmodified. The total $\mathrm{eval}$ count is $O(m)$ — one per edge for $\mathrm{sdom}$, one per bucketed vertex for $\mathrm{idom}$ — plus the linear DFS and two linear passes, with $O(n+m)$ memory.

```python
import sys


def read_graph(data):
    """Parse n, m, the start s (1-based in input), and m directed edges into a
    0-based successor list and its reverse. Returns (n, s, succ, pred)."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = int(next(it)) - 1
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for _ in range(m):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        succ[a].append(b)
        pred[b].append(a)
    return n, s, succ, pred


def dominator_tree(n, s, edges):
    """idom[v] = the immediate dominator of v for every v reachable from s;
    idom[s] = s by convention; idom[v] = -1 for v unreachable from s."""
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)

    # Depth-first search from s: preorder number dfn, order[], tree parent fa.
    dfn = [0] * n            # 1..cnt preorder number; 0 means unreached from s
    order = [0] * (n + 1)    # order[i] = the vertex whose dfn is i
    fa = [-1] * n
    cnt = 1
    dfn[s] = cnt
    order[cnt] = s
    stack = [(s, iter(succ[s]))]
    while stack:
        u, it = stack[-1]
        for w in it:
            if dfn[w] == 0:
                cnt += 1
                dfn[w] = cnt
                order[cnt] = w
                fa[w] = u
                stack.append((w, iter(succ[w])))
                break
        else:
            stack.pop()

    # Disjoint-set forest carrying the minimum-sdom witness along compressed paths.
    sdom = list(range(n))    # unprocessed vertex reads as its own sdom
    anc = [-1] * n           # forest parent; -1 while v is a root
    label = list(range(n))   # label[x] = min-sdom vertex on the compressed chain above x
    idom = [-1] * n
    bucket = [[] for _ in range(n)]

    def compress(v):
        path = []
        x = v
        while anc[x] != -1 and anc[anc[x]] != -1:
            path.append(x)
            x = anc[x]
        for x in reversed(path):
            a = anc[x]
            if dfn[sdom[label[a]]] < dfn[sdom[label[x]]]:
                label[x] = label[a]
            anc[x] = anc[a]

    def eval_(v):
        if anc[v] == -1:
            return v
        compress(v)
        return label[v]

    # Decreasing dfn: semidominators, then deferred immediate dominators.
    for i in range(cnt, 1, -1):
        v = order[i]
        for w in pred[v]:
            if dfn[w] == 0:
                continue
            u = eval_(w)
            if dfn[sdom[u]] < dfn[sdom[v]]:
                sdom[v] = sdom[u]
        bucket[sdom[v]].append(v)
        anc[v] = fa[v]
        p = fa[v]
        for x in bucket[p]:
            u = eval_(x)
            idom[x] = u if dfn[sdom[u]] < dfn[sdom[x]] else p
        bucket[p] = []

    # Increasing dfn: resolve deferred immediate dominators.
    for i in range(2, cnt + 1):
        v = order[i]
        if idom[v] != sdom[v]:
            idom[v] = idom[idom[v]]
    idom[s] = s
    return idom


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, s, succ, pred = read_graph(data)
    edges = [(a, b) for a in range(n) for b in succ[a]]
    idom = dominator_tree(n, s, edges)


if __name__ == "__main__":
    main()
```
