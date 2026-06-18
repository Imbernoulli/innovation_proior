# Dominator tree via the Lengauer-Tarjan algorithm

## Problem

In a directed flowgraph $G$ with start vertex $s$, a vertex $u$ **dominates** $v$ if every path from $s$ to $v$ passes through $u$. The **immediate dominator** $\mathrm{idom}(v)$ is the unique dominator of $v$ (other than $v$) that is dominated by all of $v$'s other dominators — the closest bottleneck above $v$. Drawing $\mathrm{idom}(v)\to v$ for every reachable $v\neq s$ yields the **dominator tree** rooted at $s$. Goal: compute $\mathrm{idom}(v)$ for all $v$ reachable from $s$, in near-linear time.

## Key idea

Deleting each vertex and re-searching ($O(nm)$), or fixpoint-intersecting dominator sets ($O(n^2)$), wastes effort on full sets when only one parent per vertex is wanted. Instead, DFS-number the graph and route everything through the **semidominator**.

**DFS numbering.** Search from $s$, giving each vertex a preorder number $\mathrm{dfn}$, a DFS tree $T$ with parent $\mathrm{fa}(v)$, and the order array. Write $u<v$ for $\mathrm{dfn}(u)<\mathrm{dfn}(v)$. Since the tree path $s\rightsquigarrow v$ is a real $s\to v$ path, $\mathrm{idom}(v)$ is a $T$-ancestor of $v$.

**Semidominator.** $\mathrm{sdom}(v)$ is the smallest-$\mathrm{dfn}$ vertex $u$ such that some path $u\to\dots\to v$ has all *interior* vertices numbered $>v$ — the highest a "detour through later-numbered vertices" can reach. Then $\mathrm{sdom}(v)<v$, $\mathrm{sdom}(v)$ is a proper $T$-ancestor of $v$, and $\mathrm{idom}(v)$ is an ancestor of $\mathrm{sdom}(v)$.

**Computing $\mathrm{sdom}$ (no path enumeration).** For $v\neq s$,
$$\mathrm{sdom}(v)=\min\Big(\{\,w:(w,v)\in E,\ w<v\,\}\ \cup\ \{\,\mathrm{sdom}(a):a>v,\ a\text{ a }T\text{-ancestor of a predecessor }w>v\text{ of }v\,\}\Big).$$
Sweep vertices in **decreasing $\mathrm{dfn}$ order**; maintain a forest of processed vertices linked to their tree parents. The query $\mathrm{eval}(w)$ returns the processed ancestor of $w$ with minimum $\mathrm{sdom}$, or $w$ itself if no link above $w$ is installed; an unprocessed smaller predecessor contributes itself because its $\mathrm{sdom}$ is still seeded to itself, while a processed root contributes its already computed $\mathrm{sdom}$. Both families collapse to one rule:
$$\mathrm{sdom}(v)=\min_{w\in\mathrm{pred}(v)}\mathrm{sdom}(\mathrm{eval}(w)).$$
$\mathrm{eval}$ is a **path-compressing** disjoint-set query: flattening the ancestor chain while carrying the minimum-$\mathrm{sdom}$ witness makes the total cost near-linear.

**Deriving $\mathrm{idom}$ from $\mathrm{sdom}$.** Let $u=\mathrm{eval}(v)$ be the vertex of minimum semidominator on the tree stretch between $\mathrm{sdom}(v)$ (exclusive) and $v$. Then
$$\mathrm{idom}(v)=\begin{cases}\mathrm{sdom}(v) & \text{if }\mathrm{sdom}(u)=\mathrm{sdom}(v),\\ \mathrm{idom}(u) & \text{otherwise.}\end{cases}$$
The second case is **deferred** (chasing the witness $u$): place $v$ in $\mathrm{bucket}(\mathrm{sdom}(v))$, drain $\mathrm{bucket}(\mathrm{fa}(v))$ right after linking $v$, recording the witness; then a single **increasing-$\mathrm{dfn}$ pass** sets $\mathrm{idom}(v)\leftarrow\mathrm{idom}(\mathrm{idom}(v))$ wherever a witness was stored.

## Algorithm

1. DFS from $s$: assign $\mathrm{dfn}$, $\mathrm{order}$, tree parent $\mathrm{fa}$. Vertices with $\mathrm{dfn}=0$ are unreachable and ignored.
2. Sweep $v$ in decreasing $\mathrm{dfn}$: $\mathrm{sdom}(v)=\min_{w\in\mathrm{pred}(v)}\mathrm{sdom}(\mathrm{eval}(w))$; push $v$ into $\mathrm{bucket}(\mathrm{sdom}(v))$; link $v$ under $\mathrm{fa}(v)$; drain $\mathrm{bucket}(\mathrm{fa}(v))$ applying the two-case rule (deferring case two).
3. Sweep $v$ in increasing $\mathrm{dfn}$: if $\mathrm{idom}(v)\neq\mathrm{sdom}(v)$ then $\mathrm{idom}(v)\leftarrow\mathrm{idom}(\mathrm{idom}(v))$. Set $\mathrm{idom}(s)=s$.

## Code

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

## Complexity

- DFS, the two sweeps, and bucketing are linear. The semidominator pass does one $\mathrm{eval}$ per edge and the idom pass one per bucketed vertex, so $O(m)$ $\mathrm{eval}$ calls. The plain link/eval version above has the classic $O(m\log n)$ near-linear bound; the balanced-link refinement improves this to $O(m\,\alpha(m,n))$.
- Memory is $O(n+m)$ — the adjacency lists plus a constant number of length-$n$ integer arrays.
- The same machinery yields the dominator tree's structure directly ($\mathrm{parent}[v]=\mathrm{idom}(v)$); subtree sizes, ancestor queries, and "how many vertices does $v$ dominate" follow from one traversal of that tree.
