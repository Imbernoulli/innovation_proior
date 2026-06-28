Let me start from the thing that actually hurts. I have a directed graph with integer weights, some negative, and I want shortest paths from a source. If all weights were non-negative I'd run Dijkstra and be done in $O(m+n\log n)$ — essentially linear. The moment one edge goes negative, Dijkstra is dead: its whole correctness rests on "extract the minimum tentative distance, it's final," and a negative edge downstream can later undercut a vertex I already finalized. So I'm thrown back on Bellman–Ford, which survives negatives but costs $O(mn)$ — quadratic on a sparse graph. Thirty years of scaling work pushed the combinatorial bound to $O(m\sqrt n\log W)$ and then it just sat there. The only people who've gotten near-linear lately did it by reformulating everything as a min-cost flow and throwing interior-point methods plus a tower of dynamic data structures at it. That feels wrong for a problem this elementary. I want to know: can I get near-linear with nothing but Dijkstra, Bellman–Ford, and graph surgery?

So what is the real object here. Johnson taught me the move: pick a price function $\phi:V\to\mathbb{Z}$ and reweight every edge as $w_\phi(u,v)=w(u,v)+\phi(u)-\phi(v)$. Along any path the intermediate $\phi$'s telescope, so $w_\phi$ of a path differs from $w$ only by the endpoints' prices, and around any *cycle* the prices cancel completely: $w_\phi(C)=w(C)$. That means $\phi$ doesn't change which paths are shortest and doesn't change which cycles are negative — $G$ and $G_\phi$ are equivalent. And if I can find a $\phi$ making *every* reduced weight $\ge 0$, then $G_\phi$ is a non-negative graph and Dijkstra finishes it. So the entire problem collapses to one thing: **find an integral $\phi$ that makes the graph non-negative** (assuming no negative cycle).

The trouble is the obvious $\phi$. Johnson's own choice is $\phi(v)=\operatorname{dist}(s,v)$; then $w_\phi(u,v)=w(u,v)+\operatorname{dist}(s,u)-\operatorname{dist}(s,v)\ge 0$ is just the triangle inequality. Beautiful, but computing those distances *is* the negative-weight SSSP problem. I've gone in a circle. I need a way to manufacture a non-negativizing $\phi$ without already knowing the distances.

Let me look harder at *why* Bellman–Ford is slow, because maybe the cost has structure I can exploit. Bellman–Ford does $n-1$ rounds of relaxing every edge. But the reason it needs many rounds isn't $n$ per se — it's that a shortest path can thread through many negative edges, and each round only pushes the frontier of correct distances forward past one "bad" segment. If a shortest path uses only a handful of negative edges, the correct distance to its endpoint ought to settle in a handful of rounds. Let me try to make that precise. For each vertex $v$, define
$$\eta_G(v) := \text{the minimum, over shortest } s\to v \text{ paths } P, \text{ of } |E^{neg}(G)\cap P|,$$
the fewest negative edges any shortest path to $v$ must use, and $\eta(G)=\max_v\eta_G(v)$. The thing I want to know is whether I can build a Dijkstra/Bellman–Ford hybrid whose cost is governed by $\sum_v\eta_G(v)$ rather than $mn$. If a hybrid like that exists, the whole game changes shape: drive $\eta$ down to something like polylog and the hybrid is near-linear.

Let me build the hybrid and see whether the cost really tracks $\eta$. Call it $\textsf{ElimNeg}$. Keep a distance estimate $d(v)$, $d(s)=0$, rest $\infty$. Alternate two phases. The **Dijkstra phase**: run Dijkstra but relax *only the non-negative edges*, with a priority queue. The **Bellman–Ford phase**: sweep once over the negative edges (it's cleanest to sweep all edges out of "marked" vertices — vertices whose label changed) and relax them, pushing any improved vertex back into the queue. Then repeat. The question that decides everything: how many alternations until $d(v)=\operatorname{dist}(s,v)$?

Before I trust an induction here, let me actually run it on a concrete graph and *watch* when each vertex settles, because if the count doesn't track $\eta$ I want to know now, not after building three more layers on top. Take $s=0$ and edges $0\!\to\!1\,(2)$, $1\!\to\!2\,(-3)$, $2\!\to\!3\,(1)$, $3\!\to\!4\,(-5)$, $4\!\to\!5\,(4)$, plus worse alternates $0\!\to\!5\,(100)$ and $2\!\to\!5\,(50)$. The two negative edges $1\!\to\!2$ and $3\!\to\!4$ both sit on the spine $0\!\to\!1\!\to\!2\!\to\!3\!\to\!4\!\to\!5$. True distances (Bellman–Ford reference): $[0,2,-1,0,-5,-1]$. Counting negative edges on each shortest path gives $\eta = [0,0,1,1,2,2]$. Now I trace the hybrid and record the first iteration in which each $d(v)$ hits its true value:

- *Iteration 0* (Dijkstra on non-negative edges only): from $0$ I reach $1$ at $2$, and the suffix $4\!\to\!5$ isn't reachable yet because $1\!\to\!2$ and $3\!\to\!4$ are negative and excluded. After this phase $d=[0,2,\infty,\infty,\infty,\infty]$. Settled correctly: $0,1$ — exactly the $\eta=0$ vertices. The BF phase then relaxes $1\!\to\!2$: $d(2)\gets 2+(-3)=-1$.
- *Iteration 1* (Dijkstra from the reactivated $2$): $2\!\to\!3$ gives $d(3)=-1+1=0$; $3\!\to\!4$ is negative so it waits. After this phase $d=[0,2,-1,0,\infty,\infty]$. Newly settled: $2,3$ — the $\eta=1$ vertices. BF relaxes $3\!\to\!4$: $d(4)\gets 0+(-5)=-5$.
- *Iteration 2*: Dijkstra from $4$ gives $d(5)=-5+4=-1$. Now $d=[0,2,-1,0,-5,-1]$, every vertex correct. Settled this round: $4,5$ — the $\eta=2$ vertices. The next BF sweep changes nothing, so it halts.

So the settling iteration per vertex came out $[0,0,1,1,2,2]$ — equal to $\eta(v)$ on the nose, and it halted after exactly $\eta(G)=2$ iterations. That's the behavior I was hoping for, and now I can see *why* the induction will work. After iteration $0$ I have every $\eta(v)=0$ vertex, by the standard Dijkstra proof on the non-negative subgraph. Suppose after iteration $i-1$ I have correct $d$ for all $v$ with $\eta(v)\le i-1$. Take a $v$ with $\eta(v)=i$ and a shortest path $P=(u_0=s,\dots,u_k=v)$ realizing it with exactly $i$ negative edges. Walk along $P$ to the *last* negative edge $(u_{j-1},u_j)$ — so the suffix $u_j\to\cdots\to v$ is all non-negative (this is exactly the $3\!\to\!4$-then-$4\!\to\!5$ shape from the trace). The prefix $s\to u_{j-1}$ uses $i-1$ negative edges, so $\eta(u_{j-1})\le i-1$ and by induction $d(u_{j-1})=\operatorname{dist}(s,u_{j-1})$ is already correct. The Bellman–Ford phase of iteration $i-1$ relaxes $(u_{j-1},u_j)$, giving $d(u_j)\le d(u_{j-1})+w(u_{j-1},u_j)=\operatorname{dist}(s,u_j)$. Then the Dijkstra phase of iteration $i$ relaxes the all-non-negative suffix in order, so $d(v)\le d(u_j)+w(\text{suffix})=\operatorname{dist}(s,v)$, and $d$ never undershoots. So $v$ is settled within $\eta(v)+1$ iterations — matching the trace, where iteration index $\eta(v)$ was the *last* one that touched $v$.

Now the running time, and I have to be careful about queue churn because that's the real cost. In the trace above each vertex entered the queue in iterations $0,\dots,\eta(v)$ and never came back after settling, and that generalizes: once $d(v)$ is final, no relaxation can lower it, so $v$ is never reactivated past iteration $\eta(v)$. Within a single iteration each vertex is added at most once in the Dijkstra phase (I'll need a small invariant for this) and at most once in the Bellman–Ford phase. So $v$ is added at most $2\eta(v)+2$ times, and the total number of queue insertions is $N\le 2\sum_v\eta_G(v)+2n$. Each insertion/extraction is $O(\log n)$. But the Bellman–Ford phase work charged to extracting $v$ is $O(\text{out-degree}(v))$, and if out-degrees are unbounded that term alone can dominate. The cleanest fix is to *require constant out-degree*. That's not a real restriction: split every high-degree vertex into a little 0-weight cycle of copies and attach one original edge per copy, which blows the graph up by only $O(m)$ vertices and edges and changes nothing about distances. With constant out-degree the BF work per extraction is $O(1)$, and the total is
$$O\big(\log n\cdot(n+\textstyle\sum_v\eta_G(v))\big).$$
So the cost does track $\sum_v\eta_G(v)$ as I'd hoped, and the constant-out-degree assumption isn't cosmetic — it's exactly what makes the per-extraction charge $O(1)$. (It also lets me write $m=\Theta(n)$ from here on.)

I owe myself the small invariant I waved at: that within one Dijkstra phase no vertex is queued twice, and more basically that I never lose an active edge. Call an edge $(v,x)$ *active* if $d(v)+w(v,x)<d(x)$. Invariant: if $(v,x)$ is active then $v$ is in the queue or marked; and if additionally $(v,x)$ is non-negative then $v$ is in the queue. This holds initially and is preserved by the three things that can disturb it — extracting a vertex (it was marked first, and all its outgoing non-negative edges were just made inactive), unmarking a vertex (only after its edges were relaxed inactive), and decreasing some $d(x)$ (we add $x$ to the queue before decreasing). From that invariant: when the algorithm halts (queue empty, all unmarked) there are no active edges, i.e. $d(v)+w(v,x)\ge d(x)$ everywhere — exactly the statement that $d$ is a valid non-negativizing price function. And one more sub-invariant for the "queued once" claim: during a Dijkstra phase, every marked vertex has $d$-value $\le$ every queued vertex's $d$-value (because we mark in increasing $d$ order and non-negative relaxations only push children above the current minimum). So a vertex, once marked, is never re-added in that phase — if it were, the new key would have to be below a marked value, contradiction. Notice this same termination argument shows that if $G$ *does* have a negative cycle, some edge stays active forever and $\textsf{ElimNeg}$ simply never halts (equivalently $\sum_v\eta_G(v)=\infty$). I'll lean on that: "doesn't terminate" is an acceptable behavior on negative-cycle inputs, and I'll convert it to a clean error later.

One bookkeeping choice pays off repeatedly: instead of the real source, attach a **dummy source** $s$ with a weight-$0$ edge to *every* vertex (and no edges in). Then $\operatorname{dist}(s,v)=\min_u\operatorname{dist}(u,v)\le 0$ for all $v$, every vertex is reachable, and $\eta$ is defined uniformly for everybody. $G_s$ has a negative cycle iff $G$ does. From now on $s$ means this dummy source unless I say otherwise.

So the architecture is clear in outline: if I can keep $\eta$ small, $\textsf{ElimNeg}$ gives me the price function in near-linear time. The whole problem has turned into **a fight to make $\eta$ small.** How?

First idea: scaling. I don't have to attack the full-precision weights at once. Goldberg's bit-scaling says: it suffices to solve the special case where weights are $\ge -1$, because then a $1$-feasible *integral* price function automatically has $w_\phi(e)\ge 0$ (integers $\ge -1$ that are $\ge -1$ after rounding are $\ge 0$), and you walk a general graph down to that case in $O(\log W)$ doublings. Let me set up the recursion concretely. I'll define a subroutine $\textsf{ScaleDown}(G,\Delta,B)$ with the contract: input weights $\ge -2B$; output an integral $\phi$ with $w_\phi(e)\ge -B$ for all $e$. If I can do that, an outer loop $\textsf{SPmain}$ that repeatedly halves $B$ (starting from $B$ around $2n$ and the weights pre-scaled up by $2n$ so everything stays integral) marches the weights from $\ge -2n$ to $\ge -1$ in $\log_2 B=O(\log n)$ rounds; then I add $+1$ to every reduced weight to clear the last $-1$'s, and since the scaling kept all distances multiples of $2n$, that $+1$ per edge can't reorder any shortest path (any two distinct path weights differ by more than $n\ge |P|$), so a final Dijkstra returns a true shortest-path tree. Each $\textsf{ScaleDown}$ call halving $B$ is the workhorse; all the difficulty lives there.

I need negativity and the $\eta$ count to talk to each other. Inside $\textsf{ScaleDown}$ I'll work mostly with the graph $G^B$: **add $B$ to every negative edge, leave non-negative edges alone.** So $w^B(e)\ge -B$ when the input had $w(e)\ge -2B$ — already that's progress on the weights. The point of shifting *only the negatives* is a bookkeeping identity: a path $P$ that uses $k$ negative edges should have $w^B(P)=w(P)+k\cdot B$, since each negative edge gains exactly $B$ and nothing else moves. Let me confirm it on the spine from my trace, $P = 0\!\to\!1\!\to\!2\!\to\!3\!\to\!4$ with weights $(2,-3,1,-5)$ — actually let me use a cleaner one, $(-4,1,-4,1)$ so $k=2$, and take $B=7$. Then $w(P)=-4+1-4+1=-6$, and shifting the two negatives gives $w^B(P)=(-4+7)+1+(-4+7)+1=3+1+3+1=8$, while $w(P)+kB=-6+2\cdot7=8$. They agree. So $w^B$ literally encodes how many negative edges a path used, scaled by $B$: $w^B(P)-w(P)$ counts negative edges times $B$. That coupling is the lever I'll keep coming back to — it turns a statement about *how negative a path is* into one about *how many negative edges it has*.

I still need to crush $\eta$. Let me think about *where* the negativity can live. Suppose I had a piece of the graph that is strongly connected and has *small diameter* — say every pair $u,v$ in it satisfies $\operatorname{dist}(u,v)\le D$ and $\operatorname{dist}(v,u)\le D$ for a smallish $D$. Could a shortest path inside such a piece be forced through many negative edges? Picture the $B$-shifted graph $G^B[V_i]$ restricted to that piece, with the dummy source. Take a shortest $s\to v$ path $P$ inside it (drop the dummy edge), starting at some $u$. If $P$ uses $k=\eta_{H^B}(v)$ negative edges, then by the identity $w^B(P)=w(P)+kB$, so $w(P)=w^B(P)-kB$. Since the dummy edges are weight $0$, $w^B$ of the path from the dummy is $\le 0$, hence $w^B(P)\le 0$, hence $w(P)\le -kB$. But $u$ and $v$ live in the same small-diameter piece, so $\operatorname{dist}(v,u)\le D$. Concatenate: I've found a closed walk of weight $w(P)+\operatorname{dist}(v,u)\le -kB+D$. If $kB>D$, that's a *negative cycle*. So with no negative cycle, $kB\le D$, i.e. $\eta_{H^B}(v)\le D/B$ inside any piece of weak diameter $D$.

I want to be sure this is a real forcing and not an artifact of my algebra, so let me push on it the other way: build a piece that *tries* to force $k=3$ negative edges and see whether the no-negative-cycle constraint really makes its diameter blow up past $D$. Take a chain $0\!\to\!1\!\to\!2\!\to\!3$ of three negative edges each of weight $-B$ with $B=5$, and add back-edges of weight $t$ to make it strongly connected. The forward walk $0\!\to\!3$ has weight $-3B=-15$, so to avoid a negative cycle the return $3\!\to\!0$ must cost $\ge 3B=15$, which means the weak diameter is at least $3B=15>2B=10$. Computing all-pairs distances: at $t=14$ (just under $3B$) there genuinely *is* a negative cycle — the construction is illegal, exactly as predicted. At $t=15$ and $t=20$ the cycle disappears and the weak diameter comes out $15$ and $20$ respectively, both $>2B$. So I cannot host three forced negative edges inside a piece of weak diameter $2B$ without creating a negative cycle — the moment I shrink the return path enough to bring the diameter down to $2B$, the cycle goes negative. The inequality isn't algebra-shuffling; the geometry actually obeys it. The content is that you can't pile up a lot of negativity in a region where everything is tightly reachable in the positive sense, because the negativity would close a loop.

This tells me exactly what diameter to aim for. I want the recursion to *halve* $\eta$: go from a problem with $\eta\le\Delta$ to subproblems with $\eta\le\Delta/2$. Set $d=\Delta/2$ and demand pieces of weak diameter $D=dB=B\Delta/2$. Then inside each piece $\eta_{H^B}(v)\le D/B=d=\Delta/2$. Recurse $\textsf{ScaleDown}$ on the union $H$ of those pieces with parameter $\Delta/2$. Recursion depth $O(\log\Delta)$, because $\Delta$ halves each level until the base case $\Delta\le 2$ where $\textsf{ElimNeg}$ alone is cheap ($\eta\le 2$).

But how do I *get* small-diameter pieces, and what does it cost? This is where I reach for low-diameter decomposition. The catch: every LDD I know of works only on **non-negative** weights, and it was built for parallel/distributed/dynamic settings, never for a sequential negative-weight shortest-path computation. Can I still use it? Yes — feed it $G^B_{\ge 0}$, the $B$-shifted graph with all *still-negative* weights rounded up to $0$. Rounding up only increases distances, so a weak-diameter bound in $G^B_{\ge 0}$ implies the same bound (or better) in $G$. The decomposition will hand me a set $E^{rem}$ of removed edges such that every strongly connected component of $G^B\setminus E^{rem}$ has weak diameter $\le D$ in $G$. That's the structure I needed.

Hold on — I haven't yet asked what I *pay* for cutting $E^{rem}$, and that's the crux of the time bound. After I make the inside-SCC edges and the between-SCC edges non-negative (I'll get to how in a second), the only possibly-negative edges left are the cut edges $E^{rem}$, and $\textsf{ElimNeg}$'s cost in the final phase is governed by how many cut edges a shortest path crosses. So I need: **for every $v$, the shortest path $P_{G^B}(v)$ crosses $O(\log^2 n)$ edges of $E^{rem}$ in expectation.** What does that demand of the decomposition? I need the cut probability of an edge to scale with its weight: $\Pr[e\in E^{rem}]=O\big(w(e)\cdot\log^2 n/D + n^{-10}\big)$. Then I can bound the expected number of cut edges on $P_{G^B}(v)$ by summing edge weights along the path. Let me check the sum works out. The total $w^B_{\ge 0}$-weight of $P_{G^B}(v)$ is at most $\eta_{G^B}(v)\cdot B$: indeed $w^B_{\ge 0}(P)\le w^B_s(P)+|P\cap E^{neg}|\cdot B$ because rounding a negative $w^B$ edge (which is $\ge -B$) up to $0$ adds at most $B$, and $w^B_s(P)\le 0$ since the dummy edge is $0$; and $|P\cap E^{neg}(G^B)|=\eta_{G^B}(v)$. So $w^B_{\ge 0}(P_{G^B}(v))\le \eta_{G^B}(v)\cdot B\le \Delta B$. Now with $D=B\Delta/2$,
$$\mathbb{E}\big[|P_{G^B}(v)\cap E^{rem}|\big]=O\Big(\frac{w^B_{\ge 0}(P_{G^B}(v))\cdot\log^2 n}{B\Delta/2}+|P|\cdot n^{-10}\Big)=O\Big(\frac{2\,\eta_{G^B}(v)\,\log^2 n}{\Delta}+n^{-9}\Big),$$
which is $O(\log^2 n)$ exactly because $\eta_{G^B}(v)\le\eta(G^B)\le\Delta$. So the single choice $D=B\Delta/2$ is doing double duty: it makes the recursion halve $\eta$ (via the diameter-forcing inequality, $\eta_{H^B}\le D/B=\Delta/2$) *and* it makes the expected cut count polylog (the $\Delta$ in $D$ cancels the $\eta\le\Delta$ in the numerator). I didn't tune two parameters to match; the same $D$ is what enters both bounds, which is reassuring — it means the two requirements aren't in tension.

So the phase structure of $\textsf{ScaleDown}$ writes itself.

**Phase 0** — decompose. Compute $E^{rem}\gets\textsf{LDD}(G^B_{\ge 0},\,D=dB)$ and let $V_1,V_2,\dots$ be the SCCs of $G^B\setminus E^{rem}$. Guarantees: each $V_i$ has weak diameter $\le dB$ (I'll verify: $\operatorname{dist}_G(u,v)\le\operatorname{dist}_{G^B_{\ge 0}}(u,v)\le dB$ for $u,v\in V_i$), and $\mathbb{E}[|P_{G^B}(v)\cap E^{rem}|]=O(\log^2 n)$ as just derived.

**Phase 1** — make the inside-SCC edges non-negative. Let $H=\bigcup_i G[V_i]$, the graph keeping only edges inside SCCs. By the small-diameter-forces-few-negatives argument above, $\eta(H^B)\le d=\Delta/2$ (no negative cycle). So $H$ meets $\textsf{ScaleDown}$'s input contract with parameter $\Delta/2$, and I recurse: $\phi_1\gets\textsf{ScaleDown}(H,\Delta/2,B)$. The stated output contract gives $w_{\phi_1}(e)\ge -B$ inside $H$... wait, I want the $B$-shifted weights non-negative, $w^B_{\phi_1}(e)\ge 0$ inside the SCCs, not just $w_{\phi_1}(e)\ge -B$. Let me re-read my own contract. The recursive call advertises $\phi_1$ with $w_{\phi_1}(e)\ge -B$ on $H$'s edges — hmm, that's not obviously non-negative once I shift by $B$. 

Let me stare at this; something's off in how I'm threading $B$. The recursion is on $H$ whose weights are the *original* $w$ restricted to inside-SCC edges, and the contract of $\textsf{ScaleDown}(H,\Delta/2,B)$ is: input $w\ge -2B$, output $\phi$ with $w_\phi(e)\ge -B$. But I want $w^B_{\phi_1}\ge 0$ on inside edges, i.e. I'm measuring against $w^B=w+B$ on the negatives. And indeed $w^B_{\phi_1}(e)=w_{\phi_1}(e)+B\ge -B+B=0$ for the negative edges, and for non-negative edges $w^B=w$ and $w_{\phi_1}(e)\ge -B$... that's still not obviously $\ge 0$. 

The externally-stated output $w_{\phi_1}(e)\ge -B$ is only the *weak* form; it can't be what I lean on, because on a non-negative edge ($w^B=w$) it gives $w^B_{\phi_1}(e)=w_{\phi_1}(e)\ge -B$, not $\ge 0$. So $w_{\phi_1}(e)\ge -B$ and $w^B_{\phi_1}(e)\ge 0$ are genuinely *not* the same statement — the latter is strictly stronger. But I don't need to derive the strong form from the weak one; I should read it off how $\textsf{ScaleDown}$ actually terminates. The last thing $\textsf{ScaleDown}(H,\Delta/2,B)$ does is run the hybrid on $H^B$ until it has no active edge, which leaves $H^B_{\phi_1}$ with *every* edge $\ge 0$, i.e. $w^B_{\phi_1}(e)\ge 0$ for all $e\in H$. The reported "$w_{\phi_1}(e)\ge -B$" is just the weaker $w_{\phi_1}=w^B_{\phi_1}-B\cdot[\text{$e$ negative}]\ge -B$ consequence I expose to callers; internally the recursion has already driven $H^B_{\phi_1}$ all the way to non-negative. So the right thing to carry out of Phase 1 is the strong internal guarantee, and indeed $G^B_{\phi_1}[V_i]$ is non-negative for every $i$. Crisis averted — I was about to lean on the weak external contract when the recursion actually hands me the strong one.

Let me also pin down the $\eta(H^B)\le d$ claim with full care, since it carries the recursion. Take $v$, let $P=P_{H^B}(v)\setminus s$ be the dummy-trimmed shortest path in $H^B_s$, first real vertex $u$. Three facts: (a) $w_{H^B}(e)=w_H(e)+B$ on negative edges; (b) the count of negative edges on $P$ equals $\eta_{H^B}(v)$; (c) $w_{H^B}(P)=w_{H^B_s}(P_{H^B}(v))\le 0$ since the dummy edge to $u$ has weight $0$ and the path is shortest from the dummy. Then
$$\operatorname{dist}_G(u,v)\le w_H(P)\overset{(a)}{\le}w_{H^B}(P)-\eta_{H^B}(v)\cdot B\overset{(c)}{\le}-\eta_{H^B}(v)\cdot B.$$
And $u,v$ in the same $V_i$ gives $\operatorname{dist}_G(v,u)\le dB$. No negative cycle $\Rightarrow\operatorname{dist}_G(u,v)+\operatorname{dist}_G(v,u)\ge 0\Rightarrow\eta_{H^B}(v)\le dB/B=d$. Clean.

**Phase 2** — make the between-SCC edges non-negative. After Phase 1 the inside-SCC edges are non-negative. Remove $E^{rem}$ and contract each $V_i$ to a node: what's left of $G^B\setminus E^{rem}$ between SCCs is a **DAG** (the $V_i$ are exactly the maximal SCCs, so no cycle survives contraction). A DAG with non-negative inside-blocks should be fixable by a single shared price per SCC in topological order: a shared price shifts all of an SCC's incoming edges by the same amount without disturbing the already-fixed inside edges, so I can lift each SCC just enough to clear its worst incoming edge. Concretely, for SCC $V_j$ let $\mu_j=\min\{w(u,v):(u,v)\in E^{neg},\,u\notin V_j,\,v\in V_j\}$ (the most negative edge entering $V_j$, or $0$), set $M_1=0$ and $M_j=M_{j-1}+\mu_j=\sum_{k\le j}\mu_k$, and let $\phi(v)=M_j$ for $v\in V_j$. For an edge $u\in V_i\to v\in V_j$ (topological order $\Rightarrow i<j$), $w_\phi(u,v)=w(u,v)+M_i-M_j=w(u,v)-\sum_{k=i+1}^{j}\mu_k\ge w(u,v)-\mu_j\ge 0$ since $\mu_j\le w(u,v)$ and the remaining $\mu_k\le 0$.

Let me check this doesn't have an off-by-one or a sign error, since prefix-sum prices are easy to get subtly wrong. Four SCCs in topo order $0,1,2,3$ with inter-SCC edges $0\!\to\!1\,(-3)$, $0\!\to\!2\,(-1)$, $1\!\to\!2\,(-5)$, $1\!\to\!3\,(-2)$, $2\!\to\!3\,(-4)$. The most-negative entering edges are $\mu=[0,-3,-5,-4]$, so the prefix sums are $M=[0,-3,-8,-12]$. Now reduce each edge by $M_u-M_v$: $0\!\to\!1$: $-3+0-(-3)=0$; $0\!\to\!2$: $-1+0-(-8)=7$; $1\!\to\!2$: $-5+(-3)-(-8)=0$; $1\!\to\!3$: $-2+(-3)-(-12)=7$; $2\!\to\!3$: $-4+(-8)-(-12)=0$. Every reduced weight is $\ge 0$, and the binding ones (each SCC's worst entering edge) land exactly at $0$ — which is the tightest the price can be without over-lifting and breaking a different edge. The formula is right. Call this $\textsf{FixDAGEdges}$; it's $O(m+n)$. Set $\phi_2=\phi_1+\psi$. Now every edge of $G^B_{\phi_2}\setminus E^{rem}$ is non-negative, so the only negatives left are inside $E^{rem}$.

**Phase 3** — clean up the cut edges with $\textsf{ElimNeg}$. The negatives of $G^B_{\phi_2}$ are all in $E^{rem}$. Run $\textsf{ElimNeg}((G^B_s)_{\phi_2},s)$ to get $\psi'$, and set $\phi_3=\phi_2+\psi'$. (A subtlety I have to respect in the order of operations: I add the dummy source to $G^B$ *first*, then apply $\phi_2$, defining $\phi_2(s)=0$, so that $(G^B_s)_{\phi_2}$ stays equivalent to $G^B_s$ and $s$ still reaches everyone. Applying the price before adding $s$ would silently change which graph I'm solving — that's a real trap, easy to get backwards.) The cost is $O((m+\sum_v\eta_{(G^B_s)_{\phi_2}}(v))\log n)$. Bound the $\eta$: a shortest $s\to v$ path $P_{G^B}(v)$ is still shortest in the equivalent $(G^B_s)_{\phi_2}$, and the negative edges it can hit there are only the cut edges, so $\eta_{(G^B_s)_{\phi_2}}(v)\le|P_{G^B}(v)\cap E^{rem}|+1$ (the $+1$ for the possibly-negative dummy edge after pricing). Taking expectations and using the Phase 0 bound, $\mathbb{E}[\eta_{(G^B_s)_{\phi_2}}(v)]\le\mathbb{E}[|P_{G^B}(v)\cap E^{rem}|]+1=O(\log^2 n)$. So Phase 3 costs $O(m\log^3 n)$ in expectation. And the output is correct: $\textsf{ElimNeg}$ makes $(G^B_s)_{\phi_3}$ non-negative, hence $w^B_{\phi_3}(e)\ge 0$, hence $w_{\phi_3}(e)\ge w^B_{\phi_3}(e)-B\ge -B$ — exactly the $\textsf{ScaleDown}$ output contract. Notice the correctness of the output never needed the no-negative-cycle assumption; if there's a cycle, $\textsf{ElimNeg}$ just loops forever, which is allowed.

Tally the recursion: Phase 0 is $O(m\log^3 n)$ (the decomposition cost — I'll verify), Phases 1–2 outside the recursive call are $O(m+n)$, Phase 3 is $O(m\log^3 n)$ expected. One recursive call per level, $O(\log\Delta)$ levels, so $\textsf{ScaleDown}=O(m\log^3 n\log\Delta)$. With $\Delta=n$, that's $O(m\log^4 n)$ per $\textsf{SPmain}$ iteration, $\times\,O(\log n)$ iterations $=O(m\log^5 n)$ expected for $\textsf{SPmain}$. 

Now I owe the decomposition itself, because everything rests on it delivering (i) weak diameter $\le D$ and (ii) $\Pr[e\in E^{rem}]=O(w(e)\log^2 n/D+n^{-10})$ in time $O(m\log^2 n+n\log^3 n)$. The existing directed decompositions were built for dynamic maintenance and are too slow / too heavy; I want a clean static one. Let me design it.

The right primitive is **ball-carving with a randomly chosen radius.** For a vertex $v$ and radius $R$, $\operatorname{Ball}^{out}(v,R)=\{u:\operatorname{dist}(v,u)\le R\}$, and its boundary $\partial$ is the edges leaving it; similarly $\operatorname{Ball}^{in}$. Why random radius? Because I want the probability an edge $(u,v)$ ends up *cut* to be proportional to its weight. If I grow a ball outward and the radius is drawn so each additional unit of radius is "one more coin flip," then $(u,v)$ is cut only if the radius lands in the window between reaching $u$ and reaching $v$, a window of width $w(u,v)$. A **geometric** radius $R\sim\textsf{Geo}(p)$ makes this exact via memorylessness: condition on $u$ already being in the ball ($R\ge\operatorname{dist}(v,u)$); then $(u,v)$ is cut iff the ball stops before reaching $v$, i.e. $R<\operatorname{dist}(v,u)+w(u,v)$, and memorylessness collapses this to $\Pr[R\le w(u,v)]\le p\cdot w(u,v)$ (the chance one of the next $w(u,v)$ coins is a head, union bound). Set $p=\min\{1,80\log n/D\}$ and the per-edge cut probability in one ball-grow comes out $O(w(e)\log n/D)$ — the shape I need, with one $\log n$; the second $\log n$ will have to come from somewhere else, presumably the recursion. And a fixed radius would not do this: it cuts a deterministic annulus and loses all proportionality to edge weight. So the geometric law isn't a convenience here — the memorylessness is exactly what makes the cut probability track $w(u,v)$ rather than the absolute position of the boundary.

Here's the decomposition. **Phase 1, mark light/heavy.** Sample $k=\Theta(\log n)$ random vertices $S$, compute their in- and out-balls of radius $D/4$, and for every $v$ read off $|\operatorname{Ball}^{in}(v,D/4)\cap S|$ and $|\operatorname{Ball}^{out}(v,D/4)\cap S|$. Mark $v$ *in-light* if its in-ball sample count is $\le 0.6k$, else *out-light* if its out-ball count is $\le 0.6k$, else *heavy*. By Chernoff, w.h.p. an in-light vertex really has $|\operatorname{Ball}^{in}(v,D/4)|\le 0.7|V|$, an out-light one $|\operatorname{Ball}^{out}(v,D/4)|\le 0.7|V|$, and a heavy one has *both* in- and out-balls $>0.5|V|$. The point of sampling is to decide cheaply, with $O(\log n)$ Dijkstras, which vertices have a small enough ball to carve.

**Phase 2, carve.** While some light vertex $v$ remains: draw $R_v\sim\textsf{Geo}(p)$, grow the appropriate ball ($\operatorname{Ball}^{in}$ if in-light, $\operatorname{Ball}^{out}$ if out-light), put its boundary edges into $E^{rem}$, **recurse** on the induced subgraph of the ball, then **delete the ball** from the working graph and continue. If ever $R_v>D/4$ or the carved ball exceeds $0.7|V|$, bail out by returning $E^{rem}=E(G)$ (singletons — trivially small diameter); I'll show this bail-out is astronomically rare. Why recurse inside the ball? Because carving guarantees the ball's *boundary* separates it from the rest (any $x$ inside, $y$ outside end up in different SCCs of $G\setminus E^{rem}$, since no edge leaves the ball after we cut $\partial$), but it says nothing about the diameter *within* the ball — the ball itself might be big and stringy. Recursing decomposes it further. And why is a light vertex's ball $\le 0.7|V|$? That's exactly what the marking guaranteed, so each recursion shrinks the vertex count to $\le 0.7|V|$, giving recursion depth $O(\log_{10/7}n)=O(\log n)$ per vertex, hence $O(n\log n)$ total recursive calls and each vertex/edge in $O(\log n)$ of them.

**Clean-up.** When no light vertices remain, only heavy ones are left, and any two heavy $x,y$ have in/out balls of radius $D/4$ each covering $>0.5|V|$, so they intersect in some $w$, giving $\operatorname{dist}(x,y)\le\operatorname{dist}(x,w)+\operatorname{dist}(w,y)\le D/4+D/4=D/2$. So I check (one Dijkstra from an arbitrary remaining vertex) that the leftover vertices have weak diameter $\le D$; if not — again astronomically rare — bail to singletons.

The diameter guarantee then follows by induction on $|V(G)|$: for any $x,y$ in the same SCC of $G\setminus E^{rem}$, either one entered a recursive ball and the other didn't (then they're separated, can't share an SCC), or both entered the same ball (induction, since the ball is $\le 0.7|V|$), or neither did (then they're among the leftover heavy vertices, covered by the clean-up check). Weak diameter $\le D$ throughout.

Runtime of the decomposition: Phase 1 is $O(\log n)$ Dijkstras $=O(|V|\log^2 n+|E|\log n)$. Phase 2's non-recursive work is $O(|V|\log n+|E|)$ (sampling each $R_v$ in $O(\log n)$, each ball by a Dijkstra charged $O(\log n)$ per explored vertex and $O(1)$ per edge, and each vertex carved at most once per level). The clean-up is one Dijkstra. Multiply by the $O(\log n)$ recursion participation per vertex/edge: total $O(|V|\log^3 n+|E|\log^2 n)$. With $m=\Theta(n)$ that's $O(m\log^3 n)$, matching what Phase 0 needed.

Now the two cut-probability pieces. The clean-up/bail-out probability: $\Pr[R_v>D/4]=(1-p)^{D/4}\le(1-80\log n/D)^{D/4}<n^{-20}$, and the Chernoff mis-marking probabilities are each $\le n^{-20}$ (deviation $0.1$ from a mean across $k=c\log n$ samples gives $e^{-2k(0.1)^2}\le n^{-20}$ for $c$ large). Union over $\le n$ vertices and $O(n\log n)$ recursive calls leaves a bail-out probability $O(n^{-18}\log n)\subseteq O(n^{-10})$ — that's the $n^{-10}$ term in the guarantee.

The boundary-cut probability is the heart, and I want it airtight, so let me do the full conditioning rather than wave at memorylessness. Fix an edge $(u,v)$ in one call (ignore the recursion for now). Order the light vertices $s_1,s_2,\dots$ in whatever order the while-loop processes them (the analysis must work for *any* order, even adversarial, because I don't control which light vertex comes next). The radii $R_1,R_2,\dots$ are the only randomness; truncate each at $R_{\max}=nW_{\max}$ (beyond which the ball can't grow anyway) so the probability space is finite and I can use the law of total probability freely. Let $B_i$ be the ball grown from $s_i$ in the graph $G_i$ remaining when $s_i$ is processed. Define event $I_i$ = "neither $u$ nor $v$ was in any earlier ball, *and* the relevant endpoint enters $B_i$" (for an out-ball that's $u\in B_i$; for an in-ball, $v\in B_i$), and $X_i$ = "the other endpoint is *excluded* from $B_i$" (out-ball: $v\notin B_i$; in-ball: $u\notin B_i$). The edge is cut at step $i$ exactly when $I_i\wedge X_i$, and the $I_i$ are disjoint (once $u$ is swallowed it's gone), so $\sum_i\Pr[I_i]\le 1$ and
$$\Pr[(u,v)\in E^{rem}]=\sum_i\Pr[I_i\wedge X_i]=\sum_i\Pr[I_i]\,\Pr[X_i\mid I_i].$$
So I just need $\Pr[X_i\mid I_i]\le p\,w(u,v)$ for each $i$. Fix the earlier radii $R_1=r_1,\dots,R_{i-1}=r_{i-1}$ so that $G_i$ contains both $u$ and $v$ (otherwise $I_i$ is false and the conditional is $0$ by convention). Now only $R_i$ is random and the graph $G_i$ is fixed. Take the out-ball case:
$$\Pr[X_i\mid I_i]=\Pr[v\notin\operatorname{Ball}^{out}(s_i,R_i)\mid u\in\operatorname{Ball}^{out}(s_i,R_i)]=\Pr[R_i<\operatorname{dist}(s_i,v)\mid R_i\ge\operatorname{dist}(s_i,u)].$$
Since $\operatorname{dist}(s_i,v)\le\operatorname{dist}(s_i,u)+w(u,v)$, this is $\le\Pr[R_i<\operatorname{dist}(s_i,u)+w(u,v)\mid R_i\ge\operatorname{dist}(s_i,u)]$, and **memorylessness** of the geometric distribution should collapse the conditioning to $\Pr[R_i\le a+b\mid R_i\ge a]\le\Pr[R_i\le b]$, killing the dependence on $a=\operatorname{dist}(s_i,u)$. This is the step the whole bound hinges on, so let me verify the collapse numerically rather than trust the slogan. With $p=0.1$ and $R\sim\textsf{Geo}(p)$ on $\{1,2,\dots\}$: for $a\in\{1,3,7,20\}$ and $b\in\{1,2,5\}$, computing $\Pr[a\le R<a+b\mid R\ge a]$ against $\Pr[R\le b]$ gives, in every case, $b=1\!:0.100=0.100$, $b=2\!:0.190=0.190$, $b=5\!:0.4095=0.4095$ — equal to five places and completely independent of $a$. So the conditioning really does collapse: the position of the boundary when it reaches $u$ doesn't matter, only the gap $w(u,v)$ does. Then $\Pr[R_i\le w(u,v)]\le p\,w(u,v)$ by the union bound over $w(u,v)$ coins. (The truncation at $R_{\max}$ is fine because $\operatorname{dist}(s_i,u)+w(u,v)\le(n-1)W_{\max}+W_{\max}=R_{\max}$, so memorylessness still applies in the truncated space.) Summing, $\Pr[(u,v)\in E^{rem}\text{ in one call}]\le p\,w(u,v)=O(w(e)\log n/D)$. Now reinstate recursion: each edge sits in $O(\log n)$ recursive calls (the participation bound), union-bound to get $O(w(e)\log^2 n/D)$, and add the $n^{-10}$ bail-out term. That matches the guarantee Phase 0 consumes, and it locates the second $\log n$ concretely — it's the recursion depth, the same $O(\log n)$ that bounded how many balls each vertex can sit inside.

Let me make sure I haven't quietly used the no-negative-cycle assumption where I shouldn't. The output correctness of $\textsf{ScaleDown}$ — that $\phi_3$ makes $w_{\phi_3}(e)\ge -B$ — holds whenever the algorithm *terminates*, with or without a cycle, because it's just the $\textsf{ElimNeg}$ guarantee plus $w^B\le w+B$. The runtime bound, and the $\eta(H^B)\le d$ and $\mathbb{E}[\eta]=O(\log^2 n)$ claims, all assumed no negative cycle. That's the right split: on a cyclic input the algorithm may run forever, and that's permitted; I'll catch it outside.

Now I have a Monte-Carlo, no-cycle-assuming engine. Two gaps remain to reach the real theorem: convert expected time to high-probability time without losing correctness, and handle the negative-cycle case by actually returning a witness cycle.

The expected-to-w.h.p. conversion is a clean bootstrap. $\textsf{SPmain}$ has expected time $\mathcal T$. Run $C\log n$ independent copies, each capped at $2\mathcal T$ steps; return the first that finishes (its output is correct by $\textsf{SPmain}$'s guarantee), else error. By Markov, each copy finishes within $2\mathcal T$ with probability $\ge 1/2$, so the chance all $C\log n$ fail is $\le 2^{-C\log n}=n^{-C}$. If the input has a negative cycle, every copy runs out of time and we correctly error. Call this $\textsf{SPMonteCarlo}$: on no-cycle inputs it returns a correct tree w.h.p.; on cyclic inputs it always errors. Time $O(m\log^6 n\log W)$ after folding in the general-weight reduction.

For the negative cycle witness, I binary-search the threshold at which the graph *just barely* stops having a negative cycle. Define $G^{+B}$ = add $B$ to *every* edge (this time including positives — a different operator from $G^B$, and I need that, because here I want to uniformly lift the whole graph until cycles turn non-negative). $\textsf{FindThresh}$ binary-searches the smallest $B\ge 0$ with no negative cycle in $G^{+B}$, using $\textsf{SPMonteCarlo}$ as the cycle detector at each probe; $O(\log W)$ probes. If the answer is $B=0$, the graph had no negative cycle and I just return the tree. If $B>0$, then $G^{+(B-1)}$ has a negative cycle but $G^{+B}$ doesn't, so a shortest-path price function $\phi$ for $G^{+B}$ (from $\textsf{SPMonteCarlo}$) makes $G^{+B}$ non-negative, and the negative cycle of $G^{+(B-1)}$ — differing by $\le 1$ per edge — survives as a *small-weight* cycle in the reweighted non-negative graph. So I pre-scale all weights by $n^3$ (forcing any genuine negative cycle to have weight $\le -n^3$, hence threshold $B\ge n^2$), keep only edges of reduced weight $\le n$, and any cycle in that low-weight subgraph is provably negative in the original: its reweighted weight is $\le n^2$ while $B|C|\ge 2B\ge 2n^2$, so $w(C)=w_{+B}(C)-B|C|\le n^2-2n^2<0$. Every branch ends by *checking* its own output (the tree is a real shortest-path tree, or the cycle is genuinely negative), restarting on the rare failure — so the whole thing is **Las Vegas**: always correct, and w.h.p. no restart, running in $O(m\log^8 n)$ after the threshold search (which dominates at $O(m\log^6 n\log^2 W)$ on the $n^3$-scaled graph), times the $\log W$ from reducing general integer weights to the $w\ge -1$ base case via Goldberg's bit-scaling and the constant-out-degree reduction.

So the causal chain, end to end: shortest paths reduce to finding a non-negativizing integral price function; the cost of finding it via a Dijkstra–Bellman-Ford hybrid is governed by $\eta$, the number of negative edges shortest paths must use; the $B$-shift identity ties negativity to negative-edge count; a low-diameter decomposition (run on the rounded non-negative graph, with geometrically random ball radii so cut probability scales with edge weight) carves the graph so that inside each small-diameter piece $\eta$ is halved — letting a scaling recursion drive $\eta$ down — while only a polylog expected number of cut edges per path survive for the hybrid to clean up; scaling halves the precision $B$ across $O(\log W)$ rounds down to the $w\ge -1$ base case where Dijkstra finishes; and a Markov bootstrap plus a threshold binary search turn the expected-time, no-cycle engine into a Las Vegas algorithm that also returns a negative cycle. Total: $O(m\log^8 n\log W)$, purely combinatorial — Dijkstra, Bellman–Ford, SCCs, topological order, and a ball-carving decomposition.

Here is the landing I commit to. The full near-linear algorithm above is the asymptotic story; for the competition deliverable I want one self-contained C++17 program that solves a single query through the same price-function skeleton — $\textsf{ElimNeg}$ to build a non-negativizing price function, then one Dijkstra — with the negative-cycle case folded in. It reads `n m s` and `m` edges `u v w` from stdin (0-indexed, weights may be negative) and prints either `NEGATIVE CYCLE` or the `n` distances $\operatorname{dist}(s,v)$ one per line (`INF` if unreachable). I build $h[v]=\operatorname{dist}(s_{\text{super}},v)\le 0$ from a virtual super-source via the marked-frontier hybrid sweep — a queue of just-changed vertices whose out-edges I relax, so a vertex settles within $\eta(v)+1$ passes — with a relaxation counter and one confirming sweep doubling as the cycle detector; then I reweight to non-negative and recover $\operatorname{dist}_G(s,v)=\operatorname{dist}_{\text{reduced}}(s,v)-h[s]+h[v]$. Everything stays in `long long` so a long chain of negative edges cannot overflow.

```cpp
// Negative-weight single-source shortest paths (combinatorial price-function method).
// Reads from stdin:  n m s   then m lines "u v w" (0-indexed vertices, integer w may be < 0);
// prints either "NEGATIVE CYCLE" if a negative cycle is reachable from the source, or n lines
// of dist(s,v) ("INF" if v is unreachable). All arithmetic in long long for overflow safety.
//
// Core idea (Johnson reweighting): find an integral price function phi with every reduced weight
// w(u,v)+phi[u]-phi[v] >= 0, then a single Dijkstra on the reweighted graph yields true distances.
// The price function is produced by ElimNeg, the Dijkstra+Bellman-Ford hybrid whose cost is governed
// by eta (the number of negative edges shortest paths must use); a super-source reaching every
// source-reachable vertex makes phi[v]=dist(s_super,v) a non-negativizing price function, and
// Bellman-Ford with an early-exit / extra relaxation round doubles as the negative-cycle detector.

#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INF = (ll)4e18;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<array<ll,3>> edges(m);                 // (u, v, w)
    vector<vector<pair<int,ll>>> adj(n);          // u -> (v, w)
    for (int i = 0; i < m; ++i) {
        ll u, v, w;
        cin >> u >> v >> w;
        edges[i] = {u, v, w};
        adj[(int)u].push_back({(int)v, w});
    }

    // Only vertices reachable from the requested source can affect its shortest-path distances, and
    // only cycles in this reachable subgraph should be reported.
    vector<char> reachable(n, 0);
    deque<int> rq;
    reachable[s] = 1;
    rq.push_back(s);
    int reachableCount = 0;
    while (!rq.empty()) {
        int u = rq.front(); rq.pop_front();
        ++reachableCount;
        for (auto [v, w] : adj[u]) {
            (void)w;
            if (!reachable[v]) {
                reachable[v] = 1;
                rq.push_back(v);
            }
        }
    }

    // ---- Step 1: ElimNeg-style price function via Bellman-Ford from a super-source ----
    // A virtual super-source has a weight-0 edge to every reachable vertex, so h[v] = dist(super, v)
    // <= 0 is defined on the reachable subgraph and is a non-negativizing price function: for any
    // reachable edge (u,v),
    // h[v] <= h[u] + w  =>  w + h[u] - h[v] >= 0.  We run the hybrid sweep: keep a queue of vertices
    // whose label changed (the "marked" frontier) and relax their out-edges; this settles a vertex
    // within eta(v)+1 rounds, so it tracks the negative-edge count rather than always doing n-1 rounds.
    vector<ll> h(n, 0);                            // h[v] starts at 0 for reachable vertices
    vector<char> inq(n, 0);
    deque<int> q;
    for (int v = 0; v < n; ++v) {
        if (reachable[v]) {
            inq[v] = 1;
            q.push_back(v);
        }
    }
    vector<int> cnt(n, 0);                         // relaxation count, for negative-cycle detection
    bool negCycle = false;

    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        inq[u] = 0;
        for (auto [v, w] : adj[u]) {
            if (!reachable[v]) continue;
            if (h[u] + w < h[v]) {
                h[v] = h[u] + w;
                if (!inq[v]) {
                    inq[v] = 1;
                    q.push_back(v);
                    if (++cnt[v] > reachableCount) { negCycle = true; break; }
                }
            }
        }
        if (negCycle) break;
    }

    // Confirm with one extra full relaxation sweep: any reachable edge still active implies a
    // negative cycle reachable from the requested source.
    if (!negCycle) {
        for (auto &e : edges) {
            int u = (int)e[0], v = (int)e[1]; ll w = e[2];
            if (reachable[u] && reachable[v] && h[u] + w < h[v]) { negCycle = true; break; }
        }
    }

    if (negCycle) {
        cout << "NEGATIVE CYCLE\n";
        return 0;
    }

    // ---- Step 2: reweight to non-negative and run Dijkstra from s ----
    // Reduced weight w + h[u] - h[v] >= 0 everywhere, so Dijkstra is correct; recover true distances
    // via dist_G(s,v) = dist_reduced(s,v) - h[s] + h[v].
    vector<ll> dr(n, INF);                         // reduced distances from s
    dr[s] = 0;
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dr[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (!reachable[v]) continue;
            ll rw = w + h[u] - h[v];              // reduced weight, >= 0
            if (dr[u] + rw < dr[v]) {
                dr[v] = dr[u] + rw;
                pq.push({dr[v], v});
            }
        }
    }

    for (int v = 0; v < n; ++v) {
        if (!reachable[v] || dr[v] >= INF) cout << "INF\n";
        else cout << (dr[v] - h[s] + h[v]) << "\n";
    }
    return 0;
}
```

The near-linear machinery I derived above — `ScaleDown`'s four phases, the geometric-radius low-diameter decomposition, `FixDAGEdges`, the scaling outer loop, and the Las Vegas/threshold wrapper — is what crushes $\eta$ to polylog and yields the $O(m\log^8 n\log W)$ bound; the single-query program keeps exactly the same price-function skeleton ($\textsf{ElimNeg}$ then Dijkstra) and the same negative-cycle semantics. Nothing beyond Dijkstra, Bellman–Ford, and price-function reweighting appears.
