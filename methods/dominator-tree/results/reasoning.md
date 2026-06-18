I have a directed graph and a start vertex $s$, and for every vertex $v$ reachable from $s$ I want its immediate dominator — the closest vertex (other than $v$ itself) that lies on *every* path from $s$ to $v$. Let me get a grip on the object before reaching for anything clever. A vertex $u$ dominates $v$ when you cannot get from $s$ to $v$ without passing through $u$. So $s$ dominates everything, $v$ dominates itself, and in between there may be a chain of dominators, each one a bottleneck strictly tighter than the last. The dominators of a single $v$ are totally ordered by "is dominated by" — if $u$ and $u'$ both dominate $v$, then on any $s\to v$ path both appear, and whichever appears first dominates the other — so they form a path $s = u_0, u_1, \dots, u_k = v$, and $\mathrm{idom}(v) = u_{k-1}$, the last one before $v$. Collecting $\mathrm{idom}(v)$ for all $v$ and drawing an edge $\mathrm{idom}(v)\to v$ gives a tree rooted at $s$ — the dominator tree — because the dominator relation is a partial order with no cycles. So the whole task is: find $\mathrm{idom}(v)$ for each $v$.

The honest first move is to read the definition literally. $u$ dominates $v$ iff deleting $u$ from the graph makes $v$ unreachable from $s$. That is checkable directly: delete $u$, run a search from $s$, see which vertices fall off. So loop over every vertex $u$, delete it, search from $s$, and every vertex that became unreachable has $u$ as one of its dominators. That correctly computes, for each $v$, its full set of dominators $\mathrm{Dom}(v)$. From the sets I can read off $\mathrm{idom}(v)$: among the strict dominators of $v$, the immediate one is the dominator that is itself dominated by all the others — equivalently the one with the largest dominator set, the deepest bottleneck. Correct, and on a small graph fine. Cost: one search per deleted vertex is $O(n + m)$, done $n$ times, so $O(n(n+m)) = O(nm)$. At $n, m$ around $10^5$ that's $10^{10}$ — hopeless. And it computes the entire $n\times n$ table of dominator sets just to extract one parent pointer per vertex; almost all of that work is thrown away.

I can do a little better without leaving the "intersection" idea. There's a fixpoint characterization: a dominator of $v$ that isn't $v$ must dominate *every* predecessor of $v$, because every $s\to v$ path enters $v$ through some predecessor. So $\mathrm{Dom}(v) = \{v\} \cup \bigcap_{u \in \mathrm{pred}(v)} \mathrm{Dom}(u)$. Iterate this to a fixpoint over all vertices, sweeping in an order where predecessors are mostly settled first (reverse postorder), and the sets converge. With bitsets each sweep is $O(n^2/w)$ and you need a few sweeps, so it's roughly $O(n^2)$ — better than $O(nm)$ but still quadratic, still carrying around full $n$-bit sets per vertex. The wastefulness is the same in spirit: I'm maintaining *sets* of dominators when I only want the *one* immediate dominator. I want to compute $\mathrm{idom}(v)$ directly, near-linear, never materializing a set.

So let me stop thinking of dominators as sets to intersect and start from structure. Run a depth-first search from $s$. That gives me a DFS tree $T$ and a *preorder number* $\mathrm{dfn}(u)$ — the order in which vertices are first reached. I'll abbreviate $u < v$ for $\mathrm{dfn}(u) < \mathrm{dfn}(v)$. Why is the DFS tree the right scaffold? Because the tree path from $s$ to $v$ in $T$ *is* one genuine $s\to v$ path in the graph, so $\mathrm{idom}(v)$ — being on every $s\to v$ path — must be one of $v$'s ancestors in $T$. That already pins $\mathrm{idom}(v)$ down to a vertical strip: somewhere on the tree path from $s$ down to $v$. Good — instead of searching all $n$ vertices for the bottleneck, I only have to locate it among the ancestors of $v$.

Now, why is $\mathrm{idom}(v)$ not simply $v$'s parent in $T$? Because there can be a non-tree edge — a forward jump or a cross edge — that sneaks into $v$'s subtree from outside, dodging the tree-path ancestors. If some edge enters the subtree below an ancestor $a$ from a vertex not under $a$, then $a$ is *not* a dominator: there's a way into $v$ avoiding $a$. So the structure I need to understand is exactly: which ancestors get bypassed by the non-tree edges, and which survive as genuine bottlenecks.

Let me think about *how* a path can reach $v$ while skipping ancestors. Picture an $s\to v$ path. It can leave the tree, wander through high-numbered vertices (things reached later in the DFS, which sit "to the right" in preorder), and re-enter near $v$. The vertices it visits in between are what let it dodge an ancestor. So for each $v$ I want to know: what is the *highest* ancestor (smallest dfn) that some path can drop down to and then reach $v$ while only ever touching vertices numbered *larger* than $v$ on the way? That "detour through the high-numbered region" is the mechanism that bypasses ancestors. Let me make it a definition and see if it's tractable.

Define, for each $v$, the smallest-dfn vertex $u$ such that there is a path $u = x_0 \to x_1 \to \dots \to x_k = v$ where every *interior* vertex $x_1, \dots, x_{k-1}$ has $\mathrm{dfn} > \mathrm{dfn}(v)$. Call it the **semidominator** $\mathrm{sdom}(v)$. The interior-vertices-are-all-bigger condition is the formal version of "the path detours only through stuff DFS reached after $v$." Why this exact shape? Because such a detour is precisely a way to arrive at $v$ from a high ancestor $u$ without being forced through the ancestors strictly between $u$ and $v$ — the interior vertices, being all larger than $v$, are never those in-between ancestors (an in-between ancestor of $v$ is smaller than $v$). So $\mathrm{sdom}(v)$ measures how high up a near-bypass can reach.

First sanity checks on this definition. The tree edge from $v$'s parent $\mathrm{fa}(v)$ to $v$ is itself a length-one path with no interior vertices, so $\mathrm{fa}(v)$ is always a candidate, and $\mathrm{fa}(v) < v$. Hence $\mathrm{sdom}(v) < v$ always — the semidominator is strictly above $v$ in number. Is it an *ancestor* of $v$ in $T$? Suppose it weren't. Then $\mathrm{sdom}(v)$ sits in some other subtree, and the defining path leaves $\mathrm{sdom}(v)$ heading toward $v$; its first step goes to a vertex that, together with the rest of the all-larger interior, would have to be reached during the DFS from $\mathrm{sdom}(v)$ *before* backtracking — which would place $v$ inside $\mathrm{sdom}(v)$'s subtree, contradicting "not an ancestor." So $\mathrm{sdom}(v)$ is a proper ancestor of $v$ in $T$. Two for two: it lives on the tree path above $v$, exactly where $\mathrm{idom}(v)$ lives.

How do $\mathrm{sdom}$ and $\mathrm{idom}$ relate? $\mathrm{sdom}(v)$ is reachable from $s$ (it's an ancestor), and from it the defining path runs to $v$ through vertices all numbered above $v$ — none of which is a strict ancestor of $v$, so none of them is forced on every path, so none of them dominates $v$. That means $\mathrm{idom}(v)$, which *does* dominate $v$, cannot be one of those interior detour vertices, and it must lie at or above $\mathrm{sdom}(v)$: $\mathrm{idom}(v)$ is an ancestor of $\mathrm{sdom}(v)$ (possibly equal). So now I have $\mathrm{idom}(v)$ squeezed between $s$ and $\mathrm{sdom}(v)$ on the tree path. If I can compute $\mathrm{sdom}$ for every vertex, I've narrowed the search enormously, and maybe $\mathrm{idom}$ falls out of $\mathrm{sdom}$ directly.

So: can I compute $\mathrm{sdom}(v)$ without enumerating all those detour paths? The definition quantifies over paths, which is the expensive part. Let me unfold one such path $v_0 \to \dots \to v_k = v$ with $v_0 = \mathrm{sdom}(v)$ and all interior $v_i > v$. Look at the last edge, $v_{k-1} \to v$. So $v_{k-1}$ is a predecessor of $v$. Two cases for that predecessor. If $k = 1$, the path is a single edge $v_0 \to v$ and $v_0$ is just a predecessor of $v$; for it to help, $v_0 < v$ (otherwise it's no smaller than $v$ and useless as a small candidate). So one family of candidates for $\mathrm{sdom}(v)$ is: **predecessors $w$ of $v$ with $w < v$.** Each such $w$ contributes itself.

If $k > 1$, the predecessor $v_{k-1}$ is an interior vertex, so $v_{k-1} > v$. Now walk back further. The whole prefix $v_0 \to \dots \to v_{k-1}$ is a path whose interior vertices are $v_1, \dots, v_{k-2}$, all greater than $v$ ... wait, that is not strong enough to make it a semidominator path for $v_{k-1}$. That definition would require the interior vertices to be greater than $v_{k-1}$, and $v_{k-1} > v$, so the bounds do not line up directly. Let me be more careful and look not at $v_{k-1}$ but at the right intermediate vertex.

Here's the cleaner way to break it. Consider a predecessor $w$ of $v$ with $w > v$ (the interior case enters through such a $w$). Take any DFS-tree ancestor $a$ of $w$ that is itself $> v$. I claim $\mathrm{sdom}(a)$ is a candidate for $\mathrm{sdom}(v)$. Why: there's a path realizing $\mathrm{sdom}(a)$ to $a$ with interior all $> a$; since $a > v$, "interior all $> a$" implies "interior all $> v$." From $a$, follow tree edges down to $w$ — those vertices are descendants of $a$ in the relevant stretch and all $> v$ as well (they're between $a$ and $w$ in the subtree, all reached after $v$). Then the single edge $w \to v$ closes it. Stitching these together: a path from $\mathrm{sdom}(a)$ to $v$ whose every interior vertex is $> v$. So $\mathrm{sdom}(a)$ qualifies as a candidate for $\mathrm{sdom}(v)$. This gives the second family: **for every predecessor $w$ of $v$ with $w > v$, and every ancestor $a$ of $w$ in $T$ with $a > v$, the value $\mathrm{sdom}(a)$.**

Do these two families *suffice* — is the true $\mathrm{sdom}(v)$ always the minimum over them? Take the actual minimizing path $v_0 \to v_1 \to \dots \to v_k = v$ with $v_0 = \mathrm{sdom}(v)$, all interior $> v$. If $k = 1$ it's a predecessor edge and the first family with $w = v_0 < v$ captures it exactly. If $k > 1$, let $j \ge 1$ be the smallest index such that $v_j$ is a tree-ancestor of $v_{k-1}$. (Such a $j$ exists: $v_{k-1}$ is its own ancestor, so $j = k-1$ works at worst.) Now I argue the prefix $v_0 \to \dots \to v_j$ is a valid semidominator path *for $v_j$*: I need every interior vertex $v_1, \dots, v_{j-1}$ to be $> v_j$. Suppose not — some $v_i$ (with $1 \le i < j$) has $v_i < v_j$. Among the original path's interior we know $v_i > v$; pick the smallest such offending $v_i$. A structural fact about DFS preorder — call it the ancestor-sandwiching property — says that on any path, if $v_i < v_j$ and both lie on this stretch with $v_i > v$, then $v_i$ must be an ancestor of $v_j$; but then $v_i$ would be an ancestor of $v_{k-1}$ too (ancestry is transitive down to $v_{k-1}$), with $i < j$, contradicting that $j$ was the *smallest* such index. So no offending $v_i$ exists, the prefix is a valid $\mathrm{sdom}$-path for $v_j$, hence $\mathrm{sdom}(v_j) \le v_0 = \mathrm{sdom}(v)$. And $v_j$ is an ancestor of $v_{k-1}$, which is a predecessor $w = v_{k-1} > v$ of $v$, with $v_j > v$. So $\mathrm{sdom}(v_j)$ is exactly a member of the second family, and it is $\le \mathrm{sdom}(v)$ — so the minimum over the two families reaches $\mathrm{sdom}(v)$. Combined with "every family member is a valid candidate, hence $\ge \mathrm{sdom}(v)$," the minimum over the two families equals $\mathrm{sdom}(v)$.

$$\mathrm{sdom}(v) = \min\Big(\{\, w : (w,v)\in E,\ w < v \,\} \ \cup\ \{\, \mathrm{sdom}(a) : a > v,\ a \text{ is a } T\text{-ancestor of some predecessor } w>v \text{ of } v \,\}\Big),$$
minimizing by dfn. This is the breakthrough: the path quantifier is gone. To compute $\mathrm{sdom}(v)$ I look at $v$'s predecessors only, and for each predecessor $w$ I need the minimum $\mathrm{sdom}(a)$ over ancestors $a$ of $w$ with $a > v$.

The second family still has an "over all ancestors with $a > v$" inside it. Here is where the processing order saves me. Process the vertices in **decreasing dfn order**: $v$ with the largest dfn first, down to $s$. When I'm about to compute $\mathrm{sdom}(v)$, every vertex with dfn $> v$ has already been processed, and its $\mathrm{sdom}$ is final. The ancestors $a > v$ of a predecessor $w$ are exactly the already-processed vertices on the tree path above $w$, down to but not including $v$ (anything $\le v$ on that path I should not consider as an interior detour). So I want: as I "activate" vertices in decreasing dfn order, maintain the forest of processed vertices linked to their tree parents, and support a query — given $w$, walk up the processed portion of $w$'s tree path and return the vertex of minimum $\mathrm{sdom}$ encountered. Call that query $\mathrm{eval}(w)$: it returns the processed ancestor $a$ of $w$ (including $w$ if $w$ is itself processed and a forest root, the boundary case) minimizing $\mathrm{sdom}(a)$ over the stretch above the current "frontier."

Concretely, I keep a disjoint-set forest. A vertex becomes part of the forest — *linked* to its tree parent — once it is processed. $\mathrm{eval}(w)$ finds the root of $w$'s current set and returns, along the path from just below the root down to $w$, the vertex with smallest $\mathrm{sdom}$. If $w$ has no installed link above it yet, $\mathrm{eval}(w) = w$. For an unprocessed smaller predecessor, that contributes $w$ itself because $\mathrm{sdom}$ is still seeded to the vertex; for a processed root, it contributes the already computed $\mathrm{sdom}(w)$. So a single uniform rule handles both families: for each predecessor $w$ of $v$, let $a = \mathrm{eval}(w)$ and offer $\mathrm{sdom}(a)$ as a candidate. Then
$$\mathrm{sdom}(v) = \min_{w \in \mathrm{pred}(v)} \mathrm{sdom}(\mathrm{eval}(w)).$$

The cost of $\mathrm{eval}$ is the cost of the path walk, and naively each walk is $O(\text{depth})$, giving $O(nm)$ again — back where I started. But this is exactly the workload a **path-compressing** disjoint-set structure was built for. When $\mathrm{eval}(w)$ walks up to the root, I splice every vertex on the walk directly to the root and, as I do, carry along the minimum-$\mathrm{sdom}$ witness: store for each vertex a $\mathrm{label}$ = the vertex of smallest $\mathrm{sdom}$ on the compressed segment above it. On the next $\mathrm{eval}$ the walk is short because the tree got flattened, and the running minimum is maintained through the labels. With path compression the total work across all $\mathrm{eval}$s is near-linear — $O(m\,\alpha(m,n))$ with the balanced-link refinement, or $O(m \log n)$ with plain compression. Each splice also updates the carried minimum, so the answer the query returns is the min over the compressed ancestor chain. That kills the bottleneck: $\mathrm{sdom}$ for all vertices in near-linear time.

Let me nail the $\mathrm{eval}$/compression bookkeeping precisely, because the carried minimum is where this gets subtle. Each vertex $x$ has an ancestor pointer $\mathrm{anc}(x)$ (its forest parent, set to its tree parent when linked) and a $\mathrm{label}(x)$. Invariant: $\mathrm{label}(x)$ is the vertex of minimum $\mathrm{sdom}$ among the vertices strictly between $x$'s set-root and $x$, inclusive of $x$. To compress, I record the current ancestor chain until the next parent is already a root, then scan that recorded path from top to bottom; if the parent's label has a smaller $\mathrm{sdom}$ than the child's label, the child adopts it, and the child is reattached one compressed jump higher. A vertex that is a forest root has no ancestor and $\mathrm{eval}$ returns the vertex itself. The comparison "smaller $\mathrm{sdom}$" is by dfn of the semidominator: $\mathrm{dfn}(\mathrm{sdom}(\cdot))$. With that, $\mathrm{eval}(w)$ returns the minimizing vertex.

So I can compute $\mathrm{sdom}(v)$ for all $v$. But I wanted $\mathrm{idom}$, and I only know $\mathrm{idom}(v)$ is somewhere between $\mathrm{sdom}(v)$ and $s$. When is $\mathrm{idom}(v)$ *equal* to $\mathrm{sdom}(v)$, and when is it strictly higher? Let me think about what could pull $\mathrm{idom}$ above $\mathrm{sdom}$. $\mathrm{sdom}(v)$ dominates $v$ exactly when no path sneaks past it; the detour that defines $\mathrm{sdom}(v)$ goes *from* $\mathrm{sdom}(v)$, so it doesn't bypass $\mathrm{sdom}(v)$ itself — but some vertex *on the tree path between $\mathrm{sdom}(v)$ and $v$* might have an even higher semidominator-reach that bypasses $\mathrm{sdom}(v)$. That's the thing to look at: the vertices strictly between $\mathrm{sdom}(v)$ and $v$ on the tree path, and their semidominators.

Let $u$ be the vertex on the tree path from $\mathrm{sdom}(v)$ (exclusive) down to $v$ with the *minimum* semidominator $\mathrm{sdom}(u)$ — found, conveniently, by exactly the same $\mathrm{eval}(v)$ query right after $v$ is linked. Two outcomes. If $\mathrm{sdom}(u) = \mathrm{sdom}(v)$ — nothing on that stretch reaches higher than $\mathrm{sdom}(v)$ does — then no path bypasses $\mathrm{sdom}(v)$ on the way to $v$, $\mathrm{sdom}(v)$ genuinely dominates $v$, and being the lowest such, $\mathrm{idom}(v) = \mathrm{sdom}(v)$. If instead $\mathrm{sdom}(u) < \mathrm{sdom}(v)$ — some vertex $u$ between them reaches *higher* than $\mathrm{sdom}(v)$ — then $\mathrm{sdom}(v)$ is bypassed (a path can drop to $\mathrm{sdom}(u)$ and re-enter below $\mathrm{sdom}(v)$), so $\mathrm{idom}(v) \ne \mathrm{sdom}(v)$; it must be higher. And here's the gift: in that case $\mathrm{idom}(v) = \mathrm{idom}(u)$. The witness $u$ — the minimal-semidominator vertex on the stretch — has the *same* immediate dominator as $v$. Intuitively, whatever bottleneck finally controls $u$ also controls $v$, because $v$ hangs below $u$ and every bypass of $v$'s region is a bypass of $u$'s. So
$$\mathrm{idom}(v) = \begin{cases} \mathrm{sdom}(v) & \text{if } \mathrm{sdom}(u) = \mathrm{sdom}(v), \\ \mathrm{idom}(u) & \text{otherwise,} \end{cases} \qquad u = \mathrm{eval}(v).$$

This is almost an algorithm, but $\mathrm{idom}(u)$ may not be known yet when I process $v$. Let me line up the timing. I'm sweeping in decreasing dfn order. When I finish $\mathrm{sdom}(v)$ I'd like to apply the rule, but $u$'s immediate dominator might still be pending. The fix is to *defer*: associate $v$ with $\mathrm{sdom}(v)$ — drop $v$ into a bucket indexed by $\mathrm{sdom}(v)$ — and only resolve $v$'s idom when the linked tree stretch below its semidominator is present. Concretely, when I link $v$ to its parent $\mathrm{fa}(v)$, I then process the bucket of $\mathrm{fa}(v)$: for each $x$ sitting in $\mathrm{bucket}(\mathrm{fa}(v))$, $\mathrm{fa}(v)$ is $\mathrm{sdom}(x)$, and now I run $u = \mathrm{eval}(x)$ over that just-linked stretch and apply the rule. If $\mathrm{sdom}(u) = \mathrm{sdom}(x)$ I can set $\mathrm{idom}(x) = \mathrm{sdom}(x)$ outright; otherwise I record $\mathrm{idom}(x) = u$ *tentatively* — a pointer to the witness whose idom equals $x$'s — to be finished later. The witness is a concrete vertex on the stretch below $\mathrm{sdom}(x)$; its own immediate dominator may still be deferred, but a later pass in increasing DFS order will finalize that witness before any vertex that points to it.

The chase is one final pass. After the decreasing sweep, walk the vertices in **increasing** dfn order. For each $v$, if the recorded $\mathrm{idom}(v)$ is not already its semidominator — meaning I stored the witness $u$, not the final answer — replace $\mathrm{idom}(v)$ by $\mathrm{idom}(\mathrm{idom}(v))$. Increasing order guarantees the witness's own idom was finalized first (the witness, being higher up the tree toward $s$ in dominator terms, gets resolved before the vertices that defer to it). One pass cleans up every deferred pointer. Then $\mathrm{idom}(s) = s$ by convention and the tree is complete.

Let me re-examine the bucket timing once, because it's the part most likely to be off by one level. I process $v$ (largest dfn first): compute $\mathrm{sdom}(v)$ from predecessors via $\mathrm{eval}$; push $v$ into $\mathrm{bucket}(\mathrm{sdom}(v))$; link $v$ under $\mathrm{fa}(v)$; then drain $\mathrm{bucket}(\mathrm{fa}(v))$, because everything whose semidominator is $\mathrm{fa}(v)$ can now have its tree stretch from $\mathrm{fa}(v)$ down evaluated (that stretch is exactly the part linked at or below $\mathrm{fa}(v)$ by now). Draining at the parent rather than at $v$ is what makes the linked region the correct one for the $\mathrm{eval}$ — the stretch from $\mathrm{sdom}(x) = \mathrm{fa}(v)$ exclusive down to $x$ is all present. Each $x$ in that bucket gets $u = \mathrm{eval}(x)$ and the two-case rule, deferred when needed. That ordering is self-consistent, and it matches what the increasing-order chase expects.

Now the running structure across one vertex: $\mathrm{sdom}$ work is one $\mathrm{eval}$ per incoming edge; idom work is one $\mathrm{eval}$ per bucket member, and each vertex is bucketed once. So total $\mathrm{eval}$ count is $O(m)$, plus the linear DFS and the two linear passes. Overall this is $O(m\,\alpha(m,n))$ with balanced linking and $O(m \log n)$ with the plain compression used here — near-linear either way, and only a handful of length-$n$ integer arrays for memory. That is the near-linear target, far from the $O(nm)$ node-deletion start.

Let me also make sure I haven't quietly assumed full reachability. Vertices unreachable from $s$ get $\mathrm{dfn} = 0$ from the DFS and simply never enter any of the passes; predecessors with $\mathrm{dfn} = 0$ are skipped when computing $\mathrm{sdom}$ (they aren't on any real $s\to v$ path). Self-loops and parallel edges are harmless — a self-loop $w\to w$ contributes $w$ as a predecessor of itself, which the formula tolerates since $\mathrm{eval}$ on the relevant region just returns a candidate no smaller than $w$; parallel edges only repeat a candidate. So the raw graph goes in unmodified.

The input wrapper stays ordinary: parse the graph, call the routine, and let the caller decide how to print the tree. Inside the routine the DFS is iterative, recording $\mathrm{dfn}$, the preorder array $\mathrm{order}$, and tree parents $\mathrm{fa}$. Then the disjoint-set forest carries $\mathrm{anc}$ (forest parent) and $\mathrm{label}$ (min-$\mathrm{sdom}$ witness), with a compressing $\mathrm{eval}$ query. The decreasing sweep computes $\mathrm{sdom}$, buckets, links, drains the parent bucket; the increasing sweep resolves deferred idoms.

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

The whole chain: deleting each vertex and re-searching costs $O(nm)$ and computes entire dominator sets only to extract one parent per vertex, so instead I DFS-number the graph and define the semidominator $\mathrm{sdom}(v)$ as the highest vertex that can reach $v$ through a detour of only-larger-numbered vertices — the formal "near-bypass of $v$'s ancestors" — which a predecessor-only formula computes as $\min_{w\in\mathrm{pred}(v)}\mathrm{sdom}(\mathrm{eval}(w))$ once vertices are swept in decreasing dfn order and $\mathrm{eval}$ is a path-compressing union–find query returning the minimum-semidominator ancestor; then the witness $u=\mathrm{eval}(v)$ on the stretch between $\mathrm{sdom}(v)$ and $v$ decides $\mathrm{idom}(v)=\mathrm{sdom}(v)$ when $\mathrm{sdom}(u)=\mathrm{sdom}(v)$ and $\mathrm{idom}(v)=\mathrm{idom}(u)$ otherwise, the latter deferred through a bucket and finished by one increasing-dfn pass — near-linear, with this plain link/eval version bounded by $O(m\log n)$ and the balanced-link refinement by $O(m\,\alpha(m,n))$.
