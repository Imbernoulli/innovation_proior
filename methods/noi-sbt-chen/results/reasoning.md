What's in front of me is a dynamic set of keys that has to do more than insert: it has to answer order statistics. Give me the $k$-th smallest key, and given a key tell me its rank — one plus how many keys sit below it — and do all of it in logarithmic time while keys keep arriving. A binary search tree is the obvious home for this, because the BST property already lays the keys out in sorted order spatially. The trouble is the one fact that ruins plain BSTs: nothing in ordinary insertion controls the height. Insert $1,2,3,\dots$ in order and every new key hangs off the rightmost node; the tree is a chain of height $n$, and every search, every insert, every rank walks the whole chain. $O(n)$. And sorted input isn't exotic — it's the common case. So the raw BST is out, and I need balance.

Let me first nail down the order-statistics part, because it tells me what every node has to carry regardless of how I balance. To find the $k$-th smallest I stand at a node $t$ and ask: how many keys are in its left subtree? Call that count, plus one for $t$ itself, the rank of $t$ inside its own subtree. If that rank equals $k$, $t$ is my answer; if $k$ is smaller I descend left; if larger I subtract off the left subtree and $t$, and descend right looking for the new $k$. Symmetrically, $\mathrm{Rank}(v)$ walks down and each time it steps right past a node it banks the whole left subtree plus that node as keys below $v$. For both I need exactly one number per node: the size of the subtree rooted there. So I'll store $s[t] = s[\mathrm{left}[t]] + s[\mathrm{right}[t]] + 1$, with $s[0]=0$ for the null subtree.

Let me make sure I have the arithmetic of $\mathrm{Select}$ right before I trust it, on a tree I can hold in my head: keys $1,2,3,4,5$ arranged with $3$ at the root, $1$ and $2$ on the left (sizes: $\mathrm{left}[\mathrm{root}]$ is the subtree $\{1,2\}$ with $s=2$), $4,5$ on the right. Call $\mathrm{Select}(4)$. At the root, the left subtree has size $2$, so the root's own rank is $2+1=3$. Since $4>3$, I subtract $3$ and go right looking for $\mathrm{Select}(1)$ in the right subtree $\{4,5\}$. There $4$ is the node with $5$ as its right child, left subtree empty, so its rank is $0+1=1$, which equals the new $k$ — I return $4$. The $4$-th smallest of $\{1,2,3,4,5\}$ is indeed $4$. And $\mathrm{Rank}(4)$: at the root $3$, since $4>3$ I bank $s[\{1,2\}]+1 = 3$ and step right; at node $4$, $4 \le 4$ so I step left into emptiness and stop; banked count $3$, plus one, gives rank $4$. Both read only $s$, so this one field carries the entire query load. It's forced on me by the queries — not optional bookkeeping, it's the thing the queries read.

Now, balance. The known schemes all work the same way at a high level: store some extra field in each node, and use it to drive a rule that fires rotations to keep the tree short. AVL stores each node's height (or just a balance factor in $\{-1,0,+1\}$) and enforces that the two subtrees of every node differ in height by at most one; after an insert you walk back up the path, and where the heights drift apart by two you rotate — a single rotation if the heavy grandchild is on the outside, a double if it's on the inside — and that restores the bound. It guarantees height about $1.44\log_2 n$. Red-black trees store a color bit and four color invariants that pin the longest root-to-leaf path to at most twice the shortest. Treaps store a random priority and keep a heap order on priorities so the tree's shape is that of a random insertion order, balanced in expectation. Splay trees store nothing extra but rotate the accessed node to the root every time, paying amortized.

Each of these solves the height problem, so why am I not just picking one? Because look at what I'd actually be carrying. For order statistics I *must* have $s[t]$ in every node. If I take AVL, I now store $s[t]$ *and* a height; the height does nothing for any query — it exists only to decide rotations. Red-black: $s[t]$ *and* a color bit, same story. Treap: $s[t]$ *and* a random priority, and now I also need a random-number source and my structure is at the mercy of the draw, which makes it harder to reason about and to debug. Splay avoids the extra field but won't give me worst-case $O(\log n)$ — a single $\mathrm{Select}$ can hit a transiently-tall tree and cost $O(n)$, and that's exactly the operation I'm building this for. So every route leaves me holding two pieces of per-node state where one of them, the balance field, is dead weight to the queries.

Stare at that asymmetry. I have a size field I'm forced to keep, and a balance field I'd rather not keep. What is the balance field really for? In AVL it's a *proxy*: height differences are a cheap stand-in for "is the tree lopsided here?" But size is a more direct measure of lopsidedness than height is. If one subtree has many more nodes than its sibling, the tree is unbalanced — that's almost the definition. So the question that won't leave me alone: can the size field I already maintain *be* the balance criterion? Could I throw the height/color/priority away entirely and let $s[t]$ both answer the queries and govern the rotations?

Let me try the most naive size rule and watch it break, so I learn what the rule has to be. Naive attempt: demand $s[\mathrm{left}[t]]$ and $s[\mathrm{right}[t]]$ never differ by more than a constant — a weight-balance condition. The problem is that after a single leaf insertion the size of a node on the path goes up by exactly one, and a one-node perturbation can tip a node from balanced to violating, and fixing it with one rotation moves a whole subtree across, which changes sizes by potentially a lot and can over-correct. Comparing a node's two *children* directly turns out to be a clumsy quantity to restore with a single local rotation, because a rotation at $t$ doesn't just swap the sizes of $t$'s two children — it relocates grandchildren. So comparing siblings is the wrong granularity. The rotation moves things at the grandchild level; the invariant should be phrased at the grandchild level too, so that the very rotation I'd fire is the thing that repairs it.

So let me phrase the condition in terms of the pieces a rotation actually shuffles. Picture a node $T$ with left child $L$ and right child $R$; let $L$ have subtrees $A$ (left) and $B$ (right), and $R$ have subtrees $C$ (left) and $D$ (right). A right rotation at $T$ lifts $L$ up and drops $T$ down to $L$'s right, carrying $B$ over to become $T$'s new left child; the players that move are $A,B,R$ and the sizes that change are those of $T$ and $L$. The grandchildren of $T$ are $A,B,C,D$. The natural thing to bound is each child's subtree against the *grandchildren on the other side* — a child versus its nephews. Concretely: don't let $R$'s subtree be smaller than either of $L$'s children, and don't let $L$'s subtree be smaller than either of $R$'s children. In symbols, for every node $t$:

$$s[\mathrm{right}[t]] \ge s[\mathrm{left}[\mathrm{left}[t]]],\ s[\mathrm{right}[\mathrm{left}[t]]]$$
$$s[\mathrm{left}[t]] \ge s[\mathrm{right}[\mathrm{right}[t]]],\ s[\mathrm{left}[\mathrm{right}[t]]].$$

Read it aloud: each child's subtree is at least as large as each of its nephew subtrees. In the $T/L/R/A/B/C/D$ picture that's exactly $s[A],s[B] \le s[R]$ and $s[C],s[D] \le s[L]$. Why is this the right granularity? Because the offending quantity after an insertion is precisely a grandchild that grew, and a single rotation at $T$ promotes a grandchild to be a child — which is the move that can fix a too-large nephew. The invariant talks about the same objects the rotation manipulates. Let me provisionally adopt this and see if (1) it forces logarithmic height and (2) it's restorable cheaply after an insert. If either fails I'll come back.

Take height first, because if this condition doesn't bound the height it's worthless no matter how cheap it is. I want the *fewest* nodes a tree of height $h$ can have while still satisfying the condition — call it $f[h]$ — because if $f[h]$ grows exponentially in $h$, then $n \ge f[h]$ forces $h = O(\log n)$. Base cases: a tree of height $0$ is a single node, $f[0]=1$; height $1$ needs at least the root and one child, $f[1]=2$. Now the inductive step, and this is where I get to use the nephew condition. Suppose $t$ roots a tree of height $h>1$. Some child of $t$ has height $h-1$ — say the left child $L$, without loss of generality — so $s[L] \ge f[h-1]$. Inside that left subtree, since $L$ has height $h-1$, *it* has a child of height $h-2$, i.e. $L$ owns a subtree of height $h-2$ and hence size at least $f[h-2]$. But that subtree of height $h-2$ is one of $L$'s children — it's a nephew of $R$ from $T$'s point of view. The condition $s[\mathrm{right}[t]] \ge s[\mathrm{right}[\mathrm{left}[t]]], s[\mathrm{left}[\mathrm{left}[t]]]$ says $R$ is at least as big as each of $L$'s children, so $s[R] \ge f[h-2]$. Now add it up:

$$f[h] = s[t] = s[L] + s[R] + 1 \ge f[h-1] + f[h-2] + 1.$$

And the bound is tight — I can build a smallest tree of each height by gluing a smallest height-$(h-1)$ tree and a smallest height-$(h-2)$ tree under a root, which satisfies the condition and has exactly $f[h-1]+f[h-2]+1$ nodes — so $f[h] = f[h-1]+f[h-2]+1$ for $h>1$. Let me just unroll the recurrence and watch the numbers, starting from $f[0]=1,f[1]=2$: $f[2]=2+1+1=4$, $f[3]=4+2+1=7$, $f[4]=7+4+1=12$, $f[5]=12+7+1=20$, $f[6]=33$, $f[7]=54$, $f[8]=88$. So the sequence is $1,2,4,7,12,20,33,54,88$. Adding one to each gives $2,3,5,8,13,21,34,55,89$ — those are exactly the Fibonacci numbers $F_3,F_4,F_5,\dots$, so I conjecture $f[h]=F_{h+3}-1$ with $F_1=F_2=1$. Check it against the recurrence: $(F_{h+2}-1)+(F_{h+1}-1)+1 = F_{h+2}+F_{h+1}-1 = F_{h+3}-1$, and the base cases $f[0]=F_3-1=2-1=1$, $f[1]=F_4-1=3-1=2$ land. So the closed form holds.

Now invert it to get height from $n$. Binet's formula gives $F_m=(\alpha^m-\beta^m)/\sqrt5$ with $\alpha=(1+\sqrt5)/2\approx1.618$ and $|\beta|<1$, so $F_m$ is within $1/2$ of $\alpha^m/\sqrt5$. From $f[h]\le n$, i.e. $n+1\ge F_{h+3}\ge \alpha^{h+3}/\sqrt5 - 1/2$, the worst height satisfies $\alpha^{h+3}/\sqrt5 \le n+1.5$, hence $h \le \log_\alpha(\sqrt5\,(n+1.5))-3 = 1.44\log_2(n+1.5)-1.33$. Let me sanity-check that this is really an upper bound and not wishful, by building the case I was most worried about — sorted input $1,2,\dots,n$ — and reading off the actual height once the structure is built. For $n=1000$ the formula caps height at $1.44\log_2(1001.5)-1.33\approx 13.0$; for $n=10000$ at $\approx17.8$. If the structure behaves, a sorted run of $1000$ keys should produce a tree no taller than about $13$ — far from the chain of height $999$ a raw BST would give. I'll hold that as a concrete prediction to test once the insertion-and-repair machinery is written; if a sorted run comes out tall, the whole size-as-balance idea is dead and I'll know it immediately. For now the algebra says the size condition alone pins the height to $O(\log n)$, with the same Fibonacci-shaped constant AVL gets — obtained without any field beyond the size.

Now the second half: after an ordinary BST insertion, how do I restore the condition, and cheaply? Insertion is the plain recursive descent — at each node on the path, increment $s[t]$ by one (since exactly one node is being added below), then recurse left or right by comparing $v$ to $\mathrm{key}[t]$, and hang the new leaf at the bottom. After that pass, which nodes can now violate? Only nodes along the insertion path saw their subtree grow, and a grandchild grew by one. So I need a repair procedure — `_rebalance(t, flag)` in code — that I run as I come back up, assuming the subtrees of $t$ are already fixed and only $t$'s own nephew condition might be off. By the left-right symmetry of the two properties I only have to think through the case where the *left* side got too heavy; the right-heavy case is the mirror image.

So suppose after the insert the left side is too big. Two sub-cases, depending on *which* of $L$'s two children is the oversized nephew. First sub-case: the outer one, $A = \mathrm{left}[\mathrm{left}[t]]$, is too big — $s[A] > s[R]$. Intuition from AVL: an outer-heavy violation wants a single rotation. Right-rotate $T$. Now $L$ rises to the top with $A$ still hanging on its left (good — $A$ was big, it belongs high), and $T$ drops to be $L$'s right child, carrying $B$ over as $T$'s left child and keeping $R$ as $T$'s right child. Draw it: new root $L$ with left $A$ and right $T$; $T$ has left $B$, right $R$. Does this restore everything? Not necessarily. $T$ now sits over $B$ and $R$, and the nephews of $T$'s subtree are now the children of $R$ — $C$ and $D$ — which might exceed $s[B]$, because $B$ used to be a grandchild and is now a child and the bound shifted. So after the rotation I have to rebalance $T$ to clean up its new neighborhood. And $L$ itself, now the root, has a right child $T$ whose subtree just got reshaped, so $L$'s right subtree changed and I should rebalance $L$ as well. One rotation, then two recursive rebalances. Fine.

Second sub-case, and this is the awkward one: the *inner* nephew $B = \mathrm{right}[\mathrm{left}[t]]$ is the offender, $s[B] > s[R]$. Now a single right-rotation at $T$ won't do it, because $B$ — the big one — would just swing from being $L$'s right child to being $T$'s left child, still buried at the same depth; rotating $T$ alone doesn't lift the *inner* grandchild. This is the AVL inner-heavy situation, which wants a double rotation. So first left-rotate $L$: that pulls $B$ up above $L$. Say $B$ had subtrees $E$ (left) and $F$ (right); after left-rotating $L$, $B$ sits where $L$ was, with $L$ as its left child (carrying $A$ and $E$) and $F$ on $B$'s right. Then right-rotate $T$: $B$ rises to the top, $L$ becomes $B$'s left child (over $A$ and $E$), and $T$ becomes $B$'s right child (over $F$ and $R$). $B$, the big inner nephew, is now the root — which is what I wanted, the inner mass lifted to the top.

Let me check this on the smallest tree that exercises the inner case, because the double rotation is exactly where it would be easy to get the relinking backwards. Insert $10$, then $5$, then $7$. After $10,5$ the tree is $10$ at the root with $5$ as its left child. Inserting $7$ goes left of $10$, then right of $5$ (since $7>5$), landing as $5$'s right child — so $7$ is the *inner* left grandchild of the root, exactly $B$, and at the root the left subtree $\{5,7\}$ has $s=2$ while the right subtree is empty with $s=0$: the inner-left nephew $s[B]=1 > s[R]=0$ fires. The repair left-rotates $5$ (lifting $7$ above it, $5$ becoming $7$'s left child) then right-rotates the root, putting $7$ on top with $5$ and $10$ as its two children. So the inner grandchild $7$ ends as the root over the two singletons $5$ and $10$ — a perfectly balanced three-node tree, which is what an inner-heavy double rotation should produce. The relinking is right.

But the two rotations have churned everything: $L$ now sits over $A,E$ and $T$ over $F,R$, and although $A,E,F,R$ are themselves still valid subtrees, the nephew conditions at $L$, at $T$, and at the new root $B$ can all be off. So I rebalance $L$ and $T$ to fix the two children, and then rebalance $B$ because even after its children are clean, $B$'s own condition might be violated. A double rotation, then three recursive rebalances.

The right-heavy situation is the exact mirror: if the outer right nephew $\mathrm{right}[\mathrm{right}[t]]$ exceeds $s[\mathrm{left}[t]]$, single left-rotate $T$; if the inner right nephew $\mathrm{left}[\mathrm{right}[t]]$ exceeds it, right-rotate $R$ then left-rotate $T$. So the full rebalance at $t$ checks four cases — outer-left, inner-left, outer-right, inner-right — fires the corresponding single or double rotation, and recursively repairs the touched nodes. The recursion makes me nervous about termination and cost, but hold that thought; let me first make sure the rotations actually keep the size field correct, because the whole edifice rests on $s$ being right.

A rotation only disturbs the sizes of the two nodes it relinks. Right-rotate at $t$: let $k = \mathrm{left}[t]$, move $\mathrm{right}[k]$ over to become $\mathrm{left}[t]$, and make $t$ the right child of $k$. After that re-linking, $k$ occupies $t$'s old position and so its subtree is exactly $t$'s old subtree — $s[k] \leftarrow s[t]$ (set this *before* recomputing $t$). Then $t$ now has whatever children it ended up with, so $s[t] \leftarrow s[\mathrm{left}[t]] + s[\mathrm{right}[t]] + 1$. Left rotation is the mirror: lift $\mathrm{right}[t]$, move its left child across, copy the old subtree size to the lifted node, and recompute $t$. Two size assignments, constant time, and every other node's size is untouched.

Now the termination and cost worry, because `_rebalance` calls itself and I have to know it doesn't spin. The honest tool here is a global potential, and the natural one for "how unbalanced is this tree" is the sum over all nodes of their depths — call it $SD$, the total depth. A short tree has small $SD$, a chain has $SD = \Theta(n^2)$; minimizing $SD$ is making the tree shallow. The claim I want is that *every rotation `_rebalance` fires strictly decreases $SD$*. Check the single-rotation case: right-rotating $T$ when $s[A] > s[R]$. The rotation lifts the nodes of $A$ up by one level each and pushes the nodes of $R$ down by one level each, with $B$'s depth unchanged and $L,T$ swapping a level. The net change in $SD$ is (nodes that went down) minus (nodes that went up) $= s[R] - s[A]$, and since $s[A] > s[R]$ this is *negative* — $SD$ drops by $s[A] - s[R] \ge 1$. The double-rotation case similarly moves the big inner subtree up two levels and the small $R$ down, and the bookkeeping gives a change of at most $s[R] - s[B] - 1 < 0$ — strictly negative again. So a rotating `_rebalance` is never gratuitous: it always buys a strict reduction in total depth. That's what makes the recursion terminate — you can't decrease a nonnegative integer potential forever.

Turn that into the amortized bound. The height is $O(\log n)$, so $SD$, the sum of $n$ depths each $O(\log n)$, is always $O(n\log n)$ — it lives in a band of width $O(n\log n)$. A single insertion adds one node at depth $O(\log n)$ and the $s$-increments don't move anyone, so each insert raises $SD$ by only $O(\log n)$ before rebalancing. Let $T$ be the total number of rotating `_rebalance` calls across a sequence of $n$ inserts. Each one drops $SD$ by at least $1$. Over $n$ inserts the total rise in $SD$ is $n \cdot O(\log n) = O(n\log n)$, and $SD$ can't go negative, so the total drop is bounded by the total rise: $T \le O(n\log n)$. Now I have to be precise about what the $O(1)$ amortized statement means. It is not one rotation per insertion; the potential only gives $O(\log n)$ rotations per insertion on average. The denominator is the number of `_rebalance` invocations. The ordinary upward pass already makes $O(n\log n)$ `_rebalance` calls over $n$ insertions, and each rotating call spawns only a constant number of follow-up calls, so the total `_rebalance` work is $O(n\log n)+O(T)=O(n\log n)$. Divided across those $O(n\log n)$ invocations, a `_rebalance` call is $O(1)$ amortized; multiplied by the $O(\log n)$ calls on an insertion path, insertion is $O(\log n)$ amortized. The structure does exactly what I needed, with logarithmic select and rank from the height bound and no field beyond the $s$ the queries already required.

There's one more thing nagging me about `_rebalance`: it's doing a lot of recursive calls and checking all four cases each time, and I suspect most of them are wasted. Let me make it leaner. When I come up from an insertion I *know* which side I descended into, so I know which property could have broken — if I went left, only the left-heavy properties (a) can fail; if I went right, only (b). So pass a flag: `_rebalance(t, False)` checks the two left-heavy cases, `_rebalance(t, True)` the two right-heavy cases. That halves the condition checks. After a rotation I still need to repair the touched nodes, and the four recursive calls that cover it are `_rebalance(left[t], False)`, `_rebalance(right[t], True)`, `_rebalance(t, False)`, `_rebalance(t, True)`.

Why those four and not all six? It's worth checking that `_rebalance(left[t], True)` and `_rebalance(right[t], False)` are genuinely unnecessary, because if I'm wrong I'll silently leave a violation — and a silently-skipped repair is the kind of bug that survives small tests and surfaces at scale. Look at the single-rotation case (the left outer one), and read off the size relations it leaves behind. Just before the rotation the left side was too heavy but only mildly — the violation was created by one insertion, so $L$'s subtree exceeds $R$'s by a bounded amount; concretely one can squeeze out $s[L] \le 2s[R] + 1$. From a `_rebalance(t, False)` check that *passed* at a node we get $s[\mathrm{left}] - 1 \le 2 s[\mathrm{right}]$, i.e. $s[\mathrm{left}] \le (2s[t]-1)/3$: a node's lighter side is at most two-thirds of its subtree. Propagate that down two levels through the double-rotation's inner node $B$ with children $E,F$: $s[B] \le (2s[L]-1)/3 \le (4s[R]+1)/3$, and then $s[E], s[F] \le (2s[B]-1)/3 \le (8s[R]+3)/9$.

The whole no-op claim now reduces to one inequality in integers: is $\lfloor(8r+3)/9\rfloor \le r$ for every $r=s[R]\ge1$? Let me just check it isn't off by one near the bottom, where ceilings bite. $r=1$: $(8+3)/9 = 11/9$, floor $1 \le 1$. $r=2$: $19/9\approx2.1$, floor $2 \le 2$. $r=3$: $27/9=3$, floor $3 \le 3$ — tight. $r=4$: $35/9\approx3.9$, floor $3 \le 4$. And asymptotically $8r/9 < r$, so once past the small cases it only gets slacker; the tight point is exactly $r=3$, and it holds. So after the right child of the rotated structure is in place, its nephews $E,F$ are already $\le s[R]$ — the would-be false-side rebalance has nothing to fix. The mirror argument with $s[A] \ge s[E]$ and $s[F] \le s[R]$ kills the other redundant call in the double case. So the two extra recursive rebalances are no-ops, and I drop them; four calls suffice.

That settles the design. The insertion is the ordinary recursive BST insert that increments $s$ on the way down, with a single line added at the bottom — rebalance the node, with the flag set by whether I went right ($v \ge \mathrm{key}[t]$) or left. Strip that one line and it's a plain BST; that's how little the balancing costs in code. Let me also pin down the two queries against the same size field. `select(k)`: at $t$, the rank of $t$ in its own subtree is $s[\mathrm{left}[t]] + 1$; if $k$ equals it return $t$, if $k$ is less go left, otherwise subtract $s[\mathrm{left}[t]]+1$ from $k$ and go right. `rank(v)`: walk down accumulating; whenever the search goes right past a node, bank $s[\mathrm{left}[t]]+1$ keys (the node's left subtree plus the node), and at the end add one to report a rank rather than a count of strictly-smaller keys.

Before I commit, let me settle the prediction I left hanging from the height proof: a sorted run of $1000$ keys should come out no taller than about $13$. Walk the very first sorted inserts to see the repair actually engage. Insert $1$: single node. Insert $2$: it's $\ge 1$ so it hangs as $1$'s right child; at the root $1$ the right nephew check has nothing below it yet, no rotation. Insert $3$: goes right of $1$, then right of $2$, landing as $2$'s right child — now at the root $1$ the right subtree $\{2,3\}$ has $s=2$ and the outer-right nephew $\mathrm{right}[\mathrm{right}[\mathrm{root}]]=3$ has $s=1 > s[\mathrm{left}[\mathrm{root}]]=0$, so the outer-right case fires: left-rotate the root, lifting $2$ to the top with $1$ and $3$ as its children. So three sorted inserts, which a plain BST would stack into a height-$2$ chain, instead produce a balanced height-$1$ tree — the chain is being broken as it forms, exactly the failure mode I started from. The formula caps the sorted-$1000$ height at $1.44\log_2(1001.5)-1.33\approx13$, and since sorted input tends to fill the tree densely rather than skirt the worst case, I'd expect the realized height to come in a bit under that, around $9$ or $10$ — and in any event nowhere near the chain of height $999$. The mechanism producing it is the same outer-rotation firing as the rightmost path keeps growing, so I'm confident the structure delivers the bound and not just the algebra; it's worth running a sorted $1$-to-$1000$ insertion once the code is down to confirm the height lands where I predict.

Let me write it out, array-based — parallel arrays $\mathrm{key}, \mathrm{left}, \mathrm{right}, s$ with index $0$ the null node and $s[0]=0$ — so the rotations and `rebalance` read exactly like the reasoning above. I land it as a single self-contained C++17 program reading from stdin: it reads a count `q` of operations, each one `I v` (insert key `v`), `S k` (print the `k`-th smallest), or `R v` (print one plus the number of stored keys strictly below `v`), and writes one line per query. Keys are 64-bit (`long long`) so nothing overflows.

The flag-directed Size Balanced Tree `_rebalance` step, especially the inner-nephew double rotation and the four follow-up repairs, is the part I would most easily get wrong under time pressure; if I were not confident I could implement it correctly within budget, I would fall back to a standard order-statistic AVL tree with subtree sizes that I have already traced as correct and ship that -- a plain correct submission beats an ambitious broken one.

```cpp
// Size Balanced Tree: an order-statistic balanced BST whose only per-node
// field, the subtree size s, both answers the queries and drives the rotations.
//
// I/O contract: reads q (number of operations); each subsequent line is an
// operation -- "I v" insert key v, "S k" print the k-th smallest stored key,
// "R v" print 1 + (number of stored keys strictly less than v). One answer per
// S/R query is written to stdout. Keys are 64-bit (long long, overflow-safe).
#include <bits/stdc++.h>
using namespace std;

// Array-based forest of nodes; index 0 is the null node with s[0] = 0.
vector<long long> key{0};                    // node key
vector<int> lc{0}, rc{0};                     // left / right child indices
vector<long long> s{0};                       // subtree size (the only extra field)
int root = 0;

int new_node(long long v) {                   // allocate a fresh leaf
    key.push_back(v); lc.push_back(0); rc.push_back(0); s.push_back(1);
    return (int)key.size() - 1;
}

int left_rotate(int t) {                      // pull right child up
    int k = rc[t];
    rc[t] = lc[k]; lc[k] = t;
    s[k] = s[t];                              // k inherits t's old subtree
    s[t] = s[lc[t]] + s[rc[t]] + 1;
    return k;
}

int right_rotate(int t) {                     // pull left child up
    int k = lc[t];
    lc[t] = rc[k]; rc[k] = t;
    s[k] = s[t];                              // k inherits t's old subtree
    s[t] = s[lc[t]] + s[rc[t]] + 1;
    return k;
}

int rebalance(int t, bool flag) {             // restore the nephew condition at t
    if (t == 0) return 0;
    if (!flag) {                              // left side may be too heavy
        if (s[lc[lc[t]]] > s[rc[t]]) {        // outer-left nephew -> single
            t = right_rotate(t);
        } else if (s[rc[lc[t]]] > s[rc[t]]) { // inner-left nephew -> double
            lc[t] = left_rotate(lc[t]); t = right_rotate(t);
        } else {
            return t;
        }
    } else {                                  // right side (mirror)
        if (s[rc[rc[t]]] > s[lc[t]]) {        // outer-right nephew -> single
            t = left_rotate(t);
        } else if (s[lc[rc[t]]] > s[lc[t]]) { // inner-right nephew -> double
            rc[t] = right_rotate(rc[t]); t = left_rotate(t);
        } else {
            return t;
        }
    }
    lc[t] = rebalance(lc[t], false);          // repair the touched nodes
    rc[t] = rebalance(rc[t], true);
    t = rebalance(t, false);
    t = rebalance(t, true);
    return t;
}

int insert_rec(int t, long long v) {          // BST insert + one rebalance
    if (t == 0) return new_node(v);
    s[t] += 1;
    if (v < key[t]) lc[t] = insert_rec(lc[t], v);
    else            rc[t] = insert_rec(rc[t], v);
    return rebalance(t, v >= key[t]);         // flag = which way we descended
}

void insert(long long v) { root = insert_rec(root, v); }

long long select_kth(long long k) {           // k-th smallest (1-indexed)
    int t = root;
    while (t) {
        long long r = s[lc[t]] + 1;           // rank of t in its own subtree
        if (k == r)      return key[t];
        else if (k < r)  t = lc[t];
        else           { k -= r; t = rc[t]; }
    }
    return 0;                                  // k out of range
}

long long rank_of(long long v) {              // 1 + #keys strictly < v
    int t = root; long long ans = 0;
    while (t) {
        if (v <= key[t]) {
            t = lc[t];
        } else {
            ans += s[lc[t]] + 1;              // bank left subtree + node
            t = rc[t];
        }
    }
    return ans + 1;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        char op; long long x;
        cin >> op >> x;
        if (op == 'I')      insert(x);
        else if (op == 'S') cout << select_kth(x) << '\n';
        else if (op == 'R') cout << rank_of(x) << '\n';
    }
    return 0;
}
```

The chain start to finish: order statistics force every node to carry its subtree size, while balance ordinarily wants a *second* field (height, color, priority) that the queries can't use — so I asked the size field to do double duty, demanding that each child's subtree be no smaller than either of its nephews, the granularity at which a single rotation can repair a violation; that condition forces a Fibonacci lower bound $f[h] = f[h-1]+f[h-2]+1$ on the node count, hence height $O(\log n)$; an ordinary BST insert then breaks the condition at one node, which a four-case `_rebalance` — a single rotation for an outer-heavy nephew, a double for an inner-heavy one, followed by recursive rebalances on the touched nodes — restores; every rotation it fires strictly lowers the total depth $SD$, and since $SD$ stays in an $O(n\log n)$ band and each insert lifts it by only $O(\log n)$, the rotations over $n$ inserts total $O(n\log n)$, making each `_rebalance` invocation $O(1)$ amortized across the $O(n\log n)$ invocations and each insertion $O(\log n)$ amortized, with logarithmic select and rank from the height bound and no field beyond the size the queries already needed.
