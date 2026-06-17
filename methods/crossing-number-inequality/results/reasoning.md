I want a lower bound on the crossing number of a graph $G$ with $n$ vertices and $m$ edges — the minimum, over all ways of drawing $G$ in the plane, of the number of points where two edges cross. Call it $\operatorname{cr}(G)$. The thing I'm trying to capture is "how unavoidably non-planar is $G$." If $G$ is planar, $\operatorname{cr}(G)=0$ and there's nothing to say; the interesting regime is when $G$ has so many edges that no drawing can keep them apart.

Let me first get a feel for the scale. How large can the crossing number be? If I just drop the $n$ vertices in general position — say on a circle — and draw every edge as a straight segment, then any two edges cross at most once, so the number of crossings is at most $\binom{m}{2} = O(m^2)$. That's a cheap upper bound, and it's surely loose, because most pairs of edges in a sensible drawing don't cross at all. But it tells me the truth lives somewhere below $m^2$, and it sets a target: a good lower bound should, for the densest graphs, climb up toward $m^2$ as well, or I'll know I'm leaving something on the table.

Now for a lower bound I need a handle on planarity, and the only genuinely topological fact I have about the plane is Euler's formula: for a connected drawing with no crossings, $V - E + F = 2$, where $F$ counts the faces including the outer one. I want to turn this into a bound on edges. In the connected simple case with at least three vertices, I can first add noncrossing edges until the graph is maximal planar; then every face is triangular, each edge is counted on two face boundaries, and $3F = 2E$. Substitute into Euler: $2 = V - E + F = V - E + \tfrac{2}{3}E = V - \tfrac{1}{3}E$, which rearranges to $E = 3V - 6$ in the maximal case, so the original connected planar graph had $E \le 3V - 6$. If the planar graph is disconnected, I can connect its components in the plane without crossings before applying the same bound; the small cases only make the weaker estimate $E \le 3V$ easier. So every simple planar graph on $V$ vertices has at most $3V$ edges. This is the rigidity I'll lean on: too many edges and the graph simply cannot be drawn flat.

How do I get from "planar graphs are sparse" to "non-planar graphs have many crossings"? Take an optimal drawing of $G$, the one achieving exactly $\operatorname{cr}(G)$ crossings. Every crossing is a point where two edges meet; if I delete one of the two edges, that crossing disappears. So by deleting at most $\operatorname{cr}(G)$ edges — one per crossing — I can destroy every crossing and be left with a drawing that has none. That surviving drawing is planar, on the same $n$ vertices, with at least $m - \operatorname{cr}(G)$ edges. But a planar graph on $n$ vertices has at most $3n$ edges, so

$$m - \operatorname{cr}(G) \le 3n,$$

which rearranges to

$$\operatorname{cr}(G) \ge m - 3n.$$

Good — that's a real lower bound, and it's not nothing: once $m$ exceeds $3n$ the graph is forced to have crossings, and the count grows with the excess. (It only bites when $m \ge 3n+1$; below that the right side is $\le 0$ and says nothing, which is fine, because a graph that sparse can be planar.)

But now compare it to my target. The cheap upper bound was $O(m^2)$, and for a dense graph — think of $m$ comparable to $n^2$, like the complete graph — this lower bound $m - 3n$ is only about $m$, linear in $m$, while the truth is somewhere up near $m^2$. The gap is enormous. For dense graphs $m - 3n$ is pathetically weak. It's the right *kind* of statement but the wrong *strength*; it gives back roughly one crossing per excess edge, whereas a dense graph ought to be drowning in crossings, quadratically many. So I have a bound whose mechanism I trust but whose magnitude is far too small. I need to amplify it.

Where is the slack? The bound came from a single, rather brutal operation: delete edges, one per crossing, until the graph goes planar. That's a blunt instrument — it throws away exactly the structure (the crossings) I'm trying to count, and the Euler constraint $3n$ caps the survivors regardless of how big $m$ is. The Euler constraint is the real bottleneck: no matter how dense $G$ is, the planar remnant can hold only $\sim 3n$ edges, so the inequality can only ever "see" an excess of $m$ over $3n$, never an excess that scales with how *many times over* the graph violates planarity. A dense graph violates planarity by a factor, and I'm only being credited for the additive overshoot.

So I want to apply the Euler constraint not once to the whole graph, but to many smaller pieces, where in each piece the constraint actually bites hard relative to the piece's size — and then somehow add up the crossings I detect across pieces. The natural way to make a smaller piece is to throw away vertices: pick a subset of vertices and keep the induced subgraph on them. If I keep a $p$-fraction of vertices, I keep roughly a $p^2$-fraction of edges (an edge survives only if both its endpoints do), but — and this is the lever — only a $p^4$-fraction of crossings (a crossing involves two edges, four vertices, all of which must survive). The crossings, the thing on the left of my inequality, get killed off *faster* than the edges as I shrink. That asymmetry between $p^2$ and $p^4$ is exactly the kind of imbalance I could try to exploit: apply the weak bound to the shrunken graph, where crossings are suppressed by $p^4$ but edges only by $p^2$, and the inequality might rearrange into something much stronger about the original $\operatorname{cr}(G)$.

But which vertices do I delete? I want to delete vertices in a way that efficiently destroys crossings while sparing edges, and there's no obvious best choice. If I try to be clever and delete a vertex that participates in many crossings, I also delete all of its incident edges, most of which had nothing to do with those crossings — I pay a large price in edges for the crossings I remove, and worse, the accounting is a tangle because I can't predict how many edges go with each deletion. Trying to hand-pick the optimal vertex set to delete looks hopeless; I'd have to be fiendishly clever and the bookkeeping is intractable.

When there's no obvious best choice and the deterministic optimization is a swamp, make the choice at random and compute on average. Keep each vertex independently with probability $p$, a parameter I'll fix later, deleting it with probability $1-p$; let $H$ be the induced subgraph on the surviving vertices. Now $H$ is random, its number of vertices, edges, and crossings all fluctuate, and any single outcome is hard to control — but I don't need a single outcome. The weak bound $\operatorname{cr}(\cdot) \ge (\text{edges}) - 3(\text{vertices})$ holds for *every* graph, hence for $H$ whatever it turns out to be:

$$\operatorname{cr}(H) \ge m_H - 3 n_H,$$

where $n_H, m_H$ are the vertex and edge counts of $H$. This is a deterministic inequality between random quantities, true for every realization. So I can take expectations of both sides, and expectations are linear and computable even when the underlying events are dependent — that's the whole point of using the first moment. Taking expectations,

$$\mathbb{E}[\operatorname{cr}(H)] \ge \mathbb{E}[m_H] - 3\,\mathbb{E}[n_H].$$

Now I just need the three expectations. The vertex count is easiest: each of the $n$ vertices survives with probability $p$ and contributes $1$ to $n_H$ when it does, so by linearity $\mathbb{E}[n_H] = pn$.

Edges next: a given edge survives in the induced subgraph exactly when both its endpoints survive, which by independence happens with probability $p \cdot p = p^2$. There are $m$ edges, so $\mathbb{E}[m_H] = p^2 m$. (The survival events of different edges aren't independent — two edges sharing a vertex are correlated — but linearity of expectation doesn't care; it sums the individual probabilities regardless.)

Now the subtle one, $\mathbb{E}[\operatorname{cr}(H)]$. The trouble is that $\operatorname{cr}(H)$ is the *minimum* over all drawings of $H$, and I have no control over what the optimal drawing of the random $H$ looks like. I can't compute the expectation of a minimum-over-redrawings directly. So let me not try to. Fix, once and for all, an optimal drawing $D$ of the original graph $G$, the one with exactly $\operatorname{cr}(G)$ crossings. When I delete vertices to form $H$, the drawing $D$ restricts to *a* drawing of $H$ — just erase the deleted vertices and their edges from the picture. This restricted drawing has some number of surviving crossings, call it $X$. It need not be the optimal drawing of $H$, so $\operatorname{cr}(H) \le X$. That inequality goes the wrong way for what I wrote above... let me check. I have $\operatorname{cr}(H) \ge m_H - 3n_H$ and $\operatorname{cr}(H) \le X$; I want to chain them into a statement about $X$, whose expectation I *can* compute. From the two, $X \ge \operatorname{cr}(H) \ge m_H - 3n_H$, so $X \ge m_H - 3n_H$ holds for every realization, and taking expectations,

$$\mathbb{E}[X] \ge \mathbb{E}[m_H] - 3\,\mathbb{E}[n_H].$$

So I should compute $\mathbb{E}[X]$, the expected number of crossings of the *fixed restricted drawing*, not of the optimal drawing of $H$. That I can do, crossing by crossing, because $X$ is just a sum over the crossings of $D$ of the indicator that the crossing survives.

A crossing in $D$ is a point where two edges cross. When does it survive into $H$? It survives iff all the vertices involved survive. How many vertices are involved? Two edges, so up to four endpoints — but if the two crossing edges shared an endpoint, they'd involve only three. Let me check whether that case can occur in an *optimal* drawing. Suppose two edges $uv$ and $uw$ share the vertex $u$ and cross at some point $x$ away from $u$. Picture the little loop: the two edges leave $u$, cross at $x$, and run on to $v$ and $w$. I can swap the two arcs between $u$ and $x$ — reroute $uv$ along the piece that was going to $w$ and vice versa up to $x$ — and this removes the crossing at $x$ without creating any new crossing, strictly lowering the count. But $D$ was optimal, so its count can't be lowered. Hence in an optimal drawing no two edges sharing a vertex cross; every crossing of $D$ is between two edges with no common endpoint, involving four *distinct* vertices.

So each crossing of $D$ involves exactly four distinct vertices, and it survives into $H$ iff all four survive, which by independence has probability $p^4$. There are $\operatorname{cr}(G)$ crossings in $D$; summing the indicators and using linearity, $\mathbb{E}[X] = p^4 \operatorname{cr}(G)$.

Now assemble. I have $\mathbb{E}[X] \ge \mathbb{E}[m_H] - 3\mathbb{E}[n_H]$, and I've computed all three:

$$p^4 \operatorname{cr}(G) \ge p^2 m - 3 p n.$$

There it is — the amplified inequality. The crossings carry $p^4$, the edges only $p^2$, exactly the asymmetry I was hunting for. Solve for $\operatorname{cr}(G)$ by dividing through by $p^4$ (which is positive):

$$\operatorname{cr}(G) \ge \frac{m}{p^2} - \frac{3n}{p^3}.$$

This holds for every $p \in (0,1]$, so I'm free to choose $p$ to make the right side as large as possible. Let me read the two terms. The first, $m/p^2$, is what I *gain* — and it grows as $p$ shrinks, which is encouraging. The second, $3n/p^3$, is what I *pay* — and it grows even faster as $p$ shrinks, with the steeper power $p^{-3}$. So I can't take $p$ too small: the penalty $p^{-3}$ eventually overruns the gain $p^{-2}$ and the right side goes negative, telling me nothing. And I can't take $p$ too large either — at $p=1$ I just recover the original weak bound $m - 3n$. The good $p$ is somewhere in between, where the gain term is large but the penalty hasn't yet swamped it.

A clean way to locate it: I want the gain $m/p^2$ to dominate the penalty $3n/p^3$ but to push $p$ as small as I can while keeping the right side comfortably positive. The two terms are in balance — comparable size — when $m/p^2 \approx 3n/p^3$, i.e. $m \approx 3n/p$, i.e. $p \approx 3n/m$. So the sweet spot has $p$ of order $n/m$. Rather than chase the exact optimum, let me pick a clean value of that order and just evaluate. Take

$$p = \frac{4n}{m}.$$

For this to be a legal probability I need $p \le 1$, i.e. $m \ge 4n$ — which is exactly the density regime I care about, the dense graphs where the weak bound failed. (And $p \le 1$ is all I need; $p=1$ at $m=4n$ is the boundary and the inequality still holds there.) Now substitute. First the gain term:

$$\frac{m}{p^2} = m \cdot \frac{m^2}{16 n^2} = \frac{m^3}{16 n^2}.$$

Then the penalty:

$$\frac{3n}{p^3} = 3n \cdot \frac{m^3}{64 n^3} = \frac{3 m^3}{64 n^2}.$$

Subtract:

$$\operatorname{cr}(G) \ge \frac{m^3}{16 n^2} - \frac{3 m^3}{64 n^2} = \frac{4 m^3 - 3 m^3}{64 n^2} = \frac{m^3}{64 n^2}.$$

So, provided $m \ge 4n$,

$$\operatorname{cr}(G) \ge \frac{m^3}{64\, n^2}.$$

Let me check this against the scale I set at the start. The cheap upper bound was $\operatorname{cr}(G) = O(m^2)$, and it's tightest for dense graphs where $m$ is of order $n^2$. Plug $m \sim n^2$ into my lower bound: $m^3/n^2 \sim n^6/n^2 = n^4 \sim m^2$. The lower bound has climbed all the way up to $m^2$, matching the upper bound up to a constant. The amplification did its job: the weak linear-in-$m$ bound has become a cubic $m^3/n^2$ that is sharp, up to the constant, for dense graphs. The cube is no accident — it's $p^{-2}$ on the edges against $p^{-4}$ on the crossings filtered through a $p \sim n/m$ choice — and that is what carries the additive bound all the way up to the quadratic ceiling.

Let me retrace the whole chain in one breath to be sure it holds. Euler's formula forces a planar graph to be sparse, $\le 3n$ edges. Deleting one edge per crossing planarizes any drawing, so $\operatorname{cr}(G) \ge m - 3n$ — true but, against the $O(m^2)$ ceiling, far too weak for dense graphs. The slack is that the Euler constraint only ever sees the additive overshoot above $3n$; to make it bite multiplicatively I shrink the graph. Keeping each vertex independently with probability $p$ suppresses vertices by $p$, edges by $p^2$, and crossings by $p^4$ — and that gap between $p^2$ and $p^4$ is the lever. Applying the weak bound to the random subgraph and taking expectations (legal because the bound holds pointwise and expectation is linear, computed on a *fixed* restricted drawing so I never face the minimum-over-redrawings) gives $p^4 \operatorname{cr}(G) \ge p^2 m - 3pn$, where the $p^4$ rests on each optimal-drawing crossing using four distinct vertices — itself forced by the swap that removes any same-vertex crossing. Dividing by $p^4$ and choosing $p = 4n/m$ to balance gain against penalty turns the weak additive bound into $\operatorname{cr}(G) \ge m^3/(64 n^2)$, valid for $m \ge 4n$, and sharp up to the constant for dense graphs.
