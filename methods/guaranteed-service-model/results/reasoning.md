OK, let me think this through from scratch. I have a supply chain — raw materials feeding components feeding subassemblies feeding final assembly feeding a distribution center feeding the customer, and in real life it branches: one component goes into several products, one assembly draws from several suppliers. Demand at the customer end is uncertain. I want to decide where in this whole web to hold safety stock, and how much, so the customer is served well and the total holding cost is as small as possible.

The thing I actually know cold is the single stage. Periodic review, base-stock policy: each period I look at demand and order back up to a level $B$. No ordering delay, so the order I place on my supplier is exactly the demand I just saw — demand flows straight up the chain and every stage ends up seeing the same end-customer demand stream. If my replenishment lead time is $L$ and I choose a demand bound $D(L)$ for that exposure window, then the least base stock that guarantees service is $B=D(L)$. With the normal-style bound $D(L)=\mu L+z\,\sigma\sqrt L$, the piece above the mean, $z\,\sigma\sqrt L$, is the safety stock. Fine. That's one location. It tells me nothing about which locations in a network should carry the buffer.

So my first instinct: just apply this formula at every stage independently. Give each stage a service level, compute $z\,\sigma\sqrt{L}$, sum it up. But that's wrong, and I can feel why before I can say it precisely. If a component stage holds a big safety stock, then the subassembly downstream of it gets replenished fast and reliably — so the subassembly needs *less* buffer. If the component holds nothing, the subassembly is exposed for longer and needs *more*. The two are coupled: my decision at one stage changes the inventory requirement at its neighbors. Treating them independently double-counts protection. And there's a circularity — how long a stage waits for replenishment depends on the supplier's safety stock, which depends on *its* supplier, all the way up. Local optimization can't see this. In real chains every function may buffer its own stage out of habit, so I need a model that can test whether the true decoupling points are only a smaller subset. I need to set inventory at all stages *jointly*.

What's the central difficulty in "jointly"? It's that "how long am I exposed" is itself a random quantity if my supplier can stock out. The honest model says: the time I wait for a replenishment is random, because sometimes my supplier is out and I have to wait for *its* replenishment. That's the Clark & Scarf (1960) picture — echelon inventory, where you track the stock at a stage plus everything downstream, and a serial system decomposes exactly into single-stage subproblems linked by induced penalty functions. It's exact and it honestly models stockouts cascading. But two things bother me for *this* purpose. The induced-cost machinery and the random delays couple the stages tightly; it's clean for a serial line and, with Rosling's transformation, for assembly, but a chain that's convergent in one place and divergent in another doesn't fold up into one tidy recursion. And the service times come out as *consequences* of the policy — I can't hand a planner a knob that says "this stage promises two days." For a strategic placement tool I'd love the service time to be a *decision*.

Let me try the other philosophy and see how far it gets. Simpson (1958) looked at the simplest network, a serial line, and asked: do I couple two adjacent operations or separate them with a buffer? His move is to have each stage *promise* a service time to its customer and *always* honor it. Stage $j$ tells its customer "any order you place at time $t$, I deliver by $t+S_j$," 100% of the time. To make that promise keepable with finite inventory, he assumes demand over any horizon is *bounded*. That last assumption is the controversial one — most of the stochastic-inventory world refuses to bound demand — so let me sit with whether it's defensible, because everything downstream rides on it.

Here's the thing. If I want a *strategic* answer to "where does the buffer go," I don't actually need to model what happens in the tail where demand blows past anything reasonable. In practice, when demand spikes far beyond the planning range, the firm doesn't just eat a cascading stockout — it expedites, runs overtime, subcontracts, pulls some extraordinary lever. So let me *define* the safety stock to cover demand up to a bound $D(\tau)$ over a horizon of $\tau$ periods, and *assume* that excess beyond the bound is absorbed by those extraordinary measures rather than propagated. With a normal demand model I'd take $D(\tau)=\tau\mu + z\,\sigma\sqrt{\tau}$ — mean demand plus the same $z\,\sigma\sqrt{\tau}$ safety term as the single-stage formula, where $z$ now encodes "how often am I willing to resort to the extraordinary lever." And notice I don't even need the demand distribution; I only need the bound function. When I ask managers for a "service level" they balk at naming a shortage cost for an external customer but are perfectly happy saying "I want 100% service over this range of demand." So the bounded-demand assumption isn't a defect — it's the modeling decision that *matches how planners think* and, crucially, it turns a stochastic problem into a deterministic one. I'll keep it.

One property I'll want for $D(\tau)$: it should be increasing in $\tau$ (longer exposure, larger possible demand) and **concave**. Concave because of pooling over time — the $\sqrt{\tau}$ already gives me that, and it'll matter later in a way I can't quite see yet, so let me just note that $D(0)=0$, $D$ increasing and concave, and move on.

Now let me build the single stage carefully inside this guaranteed-service world, because the whole network model is going to be this block repeated. Stage $j$ has a deterministic production lead time $T_j$ — the time from when all inputs are available until the finished item is ready to serve demand, including handling and transport. Two service times matter. The outbound one, $S_j$: the time $j$ promises its customer. And an inbound one, $SI_j$: the time $j$ waits to *get its inputs* and start production. If $j$'s supplier promised service time $S_i$, then $j$ can't begin until those inputs arrive, so $SI_j \ge \max\{S_i\}$ over its suppliers. So a supplier's *outbound* promise becomes the customer's *inbound* wait — that's the link that couples the network, and now it's a clean deterministic equation, not a random delay.

Let me write the inventory. At the end of period $t$, on-hand at $j$ is the base stock minus the demand that has been ordered but not yet covered out of inventory:
$$ I_j(t) = B_j - d_j(t - SI_j - T_j,\; t - S_j). $$
Why those limits? Walk it through. By the end of period $t$, stage $j$ has *replenished* inventory for all demand it observed up through $t - SI_j - T_j$ — that's how far back the pipeline reaches: it took $SI_j$ to get inputs and $T_j$ to produce. And it has *shipped* against demand observed up through $t - S_j$, because it promised to deliver within $S_j$. The difference between what it must have shipped and what it has been replenished for is the inventory it's standing on, $d_j(t-SI_j-T_j,\,t-S_j)$. So on-hand is $B_j$ minus that.

For 100% service I need $I_j(t)\ge 0$ always, i.e. $B_j \ge d_j(t-SI_j-T_j,\,t-S_j)$ for every $t$. Demand is bounded, so the worst case over that window is $D_j$ evaluated at the window length. The window length is $(t-S_j) - (t-SI_j-T_j) = SI_j + T_j - S_j$. Call that $\tau_j := SI_j + T_j - S_j$ — the **net replenishment time**. So the least base stock that guarantees service is
$$ B_j = D_j(\tau_j),\qquad \tau_j = SI_j + T_j - S_j. $$
That's the cleanest possible statement of the coupling. $\tau_j$ is the *only* window of exposure I haven't already covered with a promise: I've been replenished for everything older than $SI_j+T_j$, I've committed to ship within $S_j$, and the gap between those is what the buffer must cover. If I promise a long $S_j$ (slow service), $\tau_j$ shrinks and I need less stock; if I get fast inputs (small $SI_j$), $\tau_j$ shrinks too. And the moment I pick $S_j$, I've fixed my customer's $SI$.

Expected safety stock is $B_j$ minus expected demand over the window:
$$ E[I_j] = D_j(\tau_j) - \tau_j\,\mu_j. $$
For the normal bound that's exactly $z\,\sigma_j\sqrt{\tau_j}$ — the single-stage formula reappears, but now with the *net* replenishment time in place of the raw lead time. Good, that's the right generalization.

What about pipeline stock — the work-in-process sitting in production? That's $W_j(t) = d_j(t - SI_j - T_j,\; t - SI_j)$, i.e. $T_j$ periods of demand in process, and $E[W_j] = T_j\mu_j$. Stare at that: it depends on $T_j$ but *not* on the service times. It's a constant no matter how I set $S$ and $SI$. So when I go to choose service times to minimize cost, the pipeline term is a fixed additive constant — it drops out of the argmin. I'll carry only the safety stock $E[I_j]$ into the optimization. (I'll still report pipeline as part of total inventory, but it isn't a decision.)

Now the network. Same block at every stage, with the inbound service time of $j$ tied to the outbound service times of its suppliers. The decision variables are all the service times $\{S_j, SI_j\}$. The objective is total safety-stock holding cost. Write it out:
$$ \mathbf{P}:\quad \min \sum_j h_j\Big[ D_j(SI_j+T_j-S_j) - (SI_j+T_j-S_j)\,\mu_j \Big] $$
subject to $S_j - SI_j \le T_j$ (so $\tau_j = SI_j+T_j-S_j \ge 0$ — can't have a negative exposure window), $SI_j - S_i \ge 0$ for every arc $(i,j)$ (inbound wait at least the supplier's promise), $S_j \le s_j$ at demand nodes (marketplace cap), and $S_j, SI_j \ge 0$ and integer (periodic review, so service times are whole periods). A clean deterministic program. The question is whether I can solve it fast on a network with tens of stages.

Let me look at the objective's shape. Each term is $h_j[D_j(\tau_j) - \tau_j\mu_j]$ as a function of the service times. $\tau_j = SI_j + T_j - S_j$ is *affine* in $(SI_j, S_j)$. $D_j$ is concave and nondecreasing in its argument; the linear part $-\tau_j\mu_j$ is affine. A concave nondecreasing function composed with an affine map is concave, minus an affine term is still concave, times a positive $h_j$ is concave, and a sum of concave functions is concave. So $\mathbf{P}$ is the minimization of a **concave** function over a polyhedral feasible region. The region is convex; is it bounded? The constraints alone don't bound $S$ and $SI$ above, but I can argue optimal service times never need to exceed the sum of production lead-times — pushing a service time past that only inflates someone's $\tau$ with no benefit, because $D$ is nondecreasing — so effectively I'm over a closed bounded convex set.

Minimizing a concave function over a closed bounded convex set — the optimum is at an **extreme point** of the region. That's a real theorem (concave functions attain their minimum over a polytope at a vertex). And suddenly Simpson's all-or-nothing makes total sense as a *consequence*, not a separate fact. On a serial line with external service time zero, the extreme points are exactly the solutions where each stage has either $S_i = 0$ — hold enough safety stock to fully decouple from downstream — or $S_i = S_{i+1} + T_i$ — hold *no* safety stock and just pass your inbound-plus-lead-time straight through as your promise. Nothing in between. The concavity is the engine: because the per-stage cost curves *downward* in $\tau$ (square-root pooling), you never want an interior compromise; you want $\tau_j$ pinned to one of its extremes. That's why the buffer concentrates at a few strategic locations instead of smearing across all of them — the math *forces* concentration.

So I'm not searching a continuum. I'm searching over which extreme point. For a serial line that's a shortest-path / simple DP (Graves 1988 noticed the serial case is a shortest path), and people have done assembly networks and distribution networks separately with dynamic programming over service times (Inderfurth 1991, 1993; Inderfurth & Minner 1998; Minner 1997). But each of those handles *one* topology. My real chains are mixed: convergent here (an assembly pulling several inputs), divergent there (a component feeding several products). I want one recursion that covers all of it.

What's the most general structure for which a DP with a *single* state per stage can work? Let me think about what a DP needs. I'd process stages one at a time, and when I "finish" a subnetwork I want to summarize it to the rest of the network through as few numbers as possible. The coupling between a stage and its neighbor is a single service time. So if every subnetwork I peel off connects to the remaining network through exactly **one arc**, then I can summarize that subnetwork by a function of just *one* service time — the time on that single connecting arc. When does "exactly one connecting arc" hold? When the network is a **tree**. A spanning tree is precisely the structure where any connected piece touches the rest through one edge. Serial, assembly, and distribution are all special trees, so a tree DP would unify them. On a general acyclic network with cycles in the undirected sense, a subnetwork can connect through several arcs and I'd need several state variables — that's where tractability dies. So: restrict to spanning trees, and the state collapses to one variable per stage.

Now, *which* one variable — the inbound or the outbound service time? It depends on the direction of the single arc that connects the subnetwork to the rest. Let me set up a labeling that makes this precise. I'll relabel the nodes $1,\dots,N$ so that each node $k$ (for $k<N$) is adjacent to exactly **one** node with a higher label; call that higher-labeled neighbor $p(k)$. Can I always do this on a tree? Yes: repeatedly find an unlabeled node that is adjacent to at most one other unlabeled node and give it the next label. A tree (or any forest) always has such a node — a leaf of the remaining subgraph — so the procedure labels all $N$ nodes, and every node except the last ends up with exactly one larger-labeled neighbor. (The labeling isn't unique; that's fine.)

Define $N_k$ = the subnetwork on labels $\{1,\dots,k\}$ that is connected to $k$ — concretely $N_k = \{k\} \cup \bigcup_{i<k,(i,k)\in A} N_i \cup \bigcup_{j<k,(k,j)\in A} N_j$: node $k$ plus all the already-labeled subnetworks hanging off it. As $k$ runs $1\to N$ I'm growing the network, attaching each new node to the lower-labeled pieces it touches, and the *only* connection from $N_k$ to the not-yet-attached part runs through the single arc between $k$ and $p(k)$.

So I want, for each $k$, the minimum total safety-stock cost *inside* $N_k$, as a function of the one service time on that connecting arc. Two cases by the arc's direction:

— If $p(k)$ is downstream of $k$ (the connecting arc leaves $k$ as a supplier), then what the outside sees from $N_k$ is $k$'s *outbound* service time $S$. Define $f_k(S)$ = min cost in $N_k$ given $S_k = S$.

— If $p(k)$ is upstream of $k$ ($p(k)$ supplies $k$ through the connecting arc), then what matters is $k$'s *inbound* service time $SI$. Define $g_k(SI)$ = min cost in $N_k$ given $SI_k = SI$.

Now the recursion. The cost of $N_k$ as a function of $k$'s own inbound and outbound times $(S, SI)$ is its own safety stock plus the best the already-attached children can do, consistent with the service times $k$ exposes to them:
$$ c_k(S, SI) = h_k\big[ D_k(SI + T_k - S) - (SI + T_k - S)\mu_k \big] \;+\!\!\sum_{(i,k)\in A,\,i<k}\!\! f_i(SI) \;+\!\!\sum_{(k,j)\in A,\,j<k}\!\! g_j(S). $$
First term: $k$'s own safety-stock cost over its net replenishment time $SI+T_k-S$. Second term: for each already-attached **supplier** $i$ of $k$, the best cost of its subnetwork $N_i$ as a function of $i$'s outbound time — and $i$'s outbound time feeds $k$'s inbound, so it's evaluated at $SI$. Third term: for each already-attached **customer** $j$ of $k$, the best cost of $N_j$ as a function of $j$'s inbound time, and $j$'s inbound is fed by $k$'s outbound $S$, so it's evaluated at $S$.

Wait — let me be careful about whether I can just plug $SI$ into $f_i$ and $S$ into $g_j$, because the actual constraints are inequalities: $i$'s outbound $S_i$ must satisfy $SI \ge S_i$ (the inbound wait is *at least* the supplier's promise), and $j$'s inbound $SI_j$ must satisfy $SI_j \ge S$. So strictly I should write $\min_{S_i \le SI} f_i(S_i)$ for the supplier and $\min_{SI_j \ge S} g_j(SI_j)$ for the customer. But here's a monotonicity I can exploit. $f_i(\cdot)$ is the min cost of a subnetwork as a function of the *outbound* service its root promises — promising slower service can only *relax* the subnetwork, never make it cost more, so $f_i$ is **nonincreasing**. Therefore $\min_{S_i \le SI} f_i(S_i) = f_i(SI)$: I want the largest allowed outbound time, which is $SI$ itself. Symmetrically, $g_j(\cdot)$ as a function of the *inbound* service its root receives is **nondecreasing** — getting inputs later can only cost more — so $\min_{SI_j \ge S} g_j(SI_j) = g_j(S)$. The monotonicity collapses each min-over-the-child to a single evaluation. That's why I can write $f_i(SI)$ and $g_j(S)$ directly.

Then the two functional equations are just minimizing $c_k$ over the remaining free service time of $k$:
$$ f_k(S) = \min_{SI}\; c_k(S, SI),\qquad \max(0,\,S-T_k)\le SI \le M_k - T_k,\ SI\ \text{integer}, $$
$$ g_k(SI) = \min_{S}\; c_k(S, SI),\qquad 0 \le S \le SI + T_k,\ S\ \text{integer (and } S\le s_k \text{ at a demand node).} $$
The bound $SI \ge S - T_k$ is just $\tau_k = SI+T_k - S \ge 0$; the upper bound $M_k - T_k$ comes from the maximum replenishment time $M_k = T_k + \max\{M_i : (i,k)\in A\}$ — no point quoting an inbound wait longer than the worst case the network can actually impose. Both minimizations are over a finite integer range, so each is a plain enumeration.

The algorithm: relabel; then for $k=1$ to $N-1$, if $p(k)$ is downstream compute $f_k(S)$ for $S=0,\dots,M_k$, else compute $g_k(SI)$ for $SI=0,\dots,M_k-T_k$ (the child functions it needs are already done, since they have smaller labels). At $k=N$ compute $g_N(SI)$ over all $SI$ and take the minimum — that's the optimal cost. Then backtrack through the stored argmins to recover every $S_j, SI_j$, and from $\tau_j$ each stage's safety stock. Complexity: $N$ stages, each functional equation evaluated at up to $M$ service-time values, each requiring a min over up to $M$ values of the other service time — $O(N M^2)$, with $M$ bounded by $\sum_j T_j$. Polynomial, and for chains of 25–30 stages essentially instantaneous.

Let me double-check the one subtle thing: why does the tree give a *single* state variable and a general network not? Because in a tree, when I cut the arc between $k$ and $p(k)$, the subnetwork $N_k$ falls off as one connected piece touching the rest through that one arc — so one service time summarizes everything. In a network with an undirected cycle, cutting one arc doesn't disconnect the piece; it stays joined through other arcs, and the summary would need a service time for each of those — the state explodes. The spanning-tree restriction is exactly the condition under which the DP state stays one-dimensional. (For genuinely general acyclic networks you'd need a different, heavier approach.)

Now I want to sanity-check the bounded-demand / guaranteed-internal-service assumption itself, because I've been leaning on it hard. The worry: by *requiring* every internal stage to give 100% guaranteed service to its internal customers, am I forcing more safety stock than necessary? An internal stockout that's handled gracefully wouldn't need to be fully buffered. Let me test this on a serial line where I *relax* the internal guarantee — let internal service be whatever falls out of the base-stock choices, only forcing 100% to the *external* customer (stage 1, external service time 0), and choose base stocks to minimize total holding cost. So now I don't constrain internal service times at all.

Track net inventory and backlog. Let $Q_i(t)$ be stage $i$'s shortfall/backlog (ordered by its customer, not yet delivered), $I_i(t)$ its on-hand. With deterministic lead time $T_i$ and base stocks $B_i$,
$$ I_i(t) = [B_i - d(t-T_i,t) - Q_{i+1}(t-T_i)]^+,\qquad Q_i(t) = [d(t-T_i,t) + Q_{i+1}(t-T_i) - B_i]^+. $$
The recursion says: my backlog grows when demand plus my supplier's backlog (the inputs I'm still waiting on) exceeds my base stock. Unrolling by induction,
$$ Q_i(t) = \max\Big[0,\ d(t-T_i,t)-B_i,\ d(t-T_i-T_{i+1},t)-B_i-B_{i+1},\ \dots,\ d(t-T_i-\cdots-T_N,t)-B_i-\cdots-B_N\Big]. $$
To give the external customer 100% service I need $Q_1(t)=0$ always, which (demand bounded by $D$) holds iff $B_1+\cdots+B_i \ge D(T_1+\cdots+T_i)$ for all $i$. Call these constraints (A3): cumulative base stock must cover cumulative-lead-time demand.

Net inventory holding cost: $\sum_i h_i E[I_i] = \sum_i h_i (B_i - \mu T_i + E[Q_{i+1}(t-T_i)] - E[Q_i(t)])$. Introduce echelon holding costs $e_i = h_i - h_{i+1}$ (cost to move a unit from stage $i+1$ to $i$). Dropping the constant $-\mu T_i$ terms and using $Q_1\equiv 0$, the problem becomes
$$ \mathbf{P^*}:\quad \min \sum_i h_i B_i - \sum_{i\ge 2} e_{i-1} E[Q_i] \quad\text{s.t.}\quad B_1+\cdots+B_i \ge D(T_1+\cdots+T_i),\ B_i\ge 0. $$
Claim: if echelon holding costs are nonnegative ($h_i \ge h_{i+1}$, cost accrues going downstream) and $D$ is nondecreasing, the optimum has *all* constraints (A3) **binding**, giving
$$ B_1 = D(T_1),\quad B_i = D(T_1+\cdots+T_i) - D(T_1+\cdots+T_{i-1}). $$
Proof by an exchange argument. Suppose some constraint $k<N$ is slack: $B_1^*+\cdots+B_k^* > D(T_1+\cdots+T_k)$, with the first $k-1$ tight. Build a new solution that makes constraint $k$ tight by shaving $\Delta = B_k^* - [D(\sum_1^k T_i) - D(\sum_1^{k-1}T_i)] > 0$ off $B_k$ and adding it to $B_{k+1}$: $B_k^{**}=B_k^*-\Delta$, $B_{k+1}^{**}=B_{k+1}^*+\Delta$, all others unchanged. Feasibility holds — cumulative sums up through $k-1$ unchanged, the sum through $k$ now exactly meets $D$, sums through $k+1$ and beyond unchanged, and $B_k^{**}\ge 0$ since $D$ nondecreasing makes $\Delta \le B_k^*$. Compare objectives. The linear base-stock part changes by
$$
\sum_i h_i B_i^{**}-\sum_i h_i B_i^* = (-h_k+h_{k+1})\Delta = -e_k\Delta.
$$
Now look at the backlog part from the unrolled max expression for $Q_i$. For stages farther upstream than $k+1$, $i>k+1$, the cumulative base stocks appearing in $Q_i$ are unchanged, so $E[Q_i]^{**}=E[Q_i]^*$. For stages at or downstream of $k$, $i<k+1$, the relevant cumulative base stocks can be smaller by at most $\Delta$, so $E[Q_i]^*\le E[Q_i]^{**}\le E[Q_i]^*+\Delta$. At stage $k+1$, the added base stock can reduce backlog by at most $\Delta$, so $E[Q_{k+1}]^*\ge E[Q_{k+1}]^{**}\ge E[Q_{k+1}]^*-\Delta$. The objective contains backlog with a minus sign:
$$
-\sum_{i=2}^N e_{i-1}E[Q_i].
$$
Because every echelon cost is nonnegative, the possible increases in $E[Q_i]$ for $i<k+1$ can only reduce this part of the objective, and the only possible increase comes from reducing $Q_{k+1}$; that increase is at most $e_k\Delta$. Therefore
$$
-\sum_{i=2}^N e_{i-1}E[Q_i]^{**}\le -\sum_{i=2}^N e_{i-1}E[Q_i]^*+e_k\Delta.
$$
The possible $+e_k\Delta$ in the backlog term cancels the $-e_k\Delta$ in the base-stock term, so the total objective does not increase. So the new solution is feasible and no worse, and now one more cumulative constraint binds. If the first slack constraint is $k=N$, I just reduce $B_N$ until the last cumulative constraint binds; there is no $B_{N+1}$ to receive mass, and the same signs make the objective no larger. Repeat the exchange until every cumulative constraint is binding; the binding equations force the unique solution
$$
B_1=D(T_1),\qquad B_i=D(T_1+\cdots+T_i)-D(T_1+\cdots+T_{i-1}).
$$
This is exactly what I needed: once internal service guarantees are relaxed in a serial line, the optimal cumulative base stock is pinned by the demand bounds, not by the holding costs except for the nonnegative-echelon-cost condition. The result extends to assembly systems through Rosling's 1989 transformation to an equivalent serial system.

Notice the optimal base stocks here *don't depend on the holding costs at all* — only on the lead times and the demand bound. That gives me a validation I should run rather than an assumption: solve matched serial instances with guaranteed internal service and with this relaxed $\mathbf{P^*}$, using the same Poisson demand bounds, lead-time shapes, and holding-cost shapes, and measure the cost of the guarantee. The proof tells me exactly what the relaxed side should be, so the comparison is well defined. For the model itself I do not need that outcome yet; the math has already isolated the tradeoff. I'll keep the guaranteed-service formulation as the model and the tree DP as the solver, with that comparison as a check on the approximation.

Let me also pin down one piece I waved at: combining demand for a stage with multiple successors. A component feeding several products needs a single demand bound to position its safety stock. If I just sum the downstream bounds I'm assuming no pooling across products — pessimistic. The general form for the upstream stage $i$ feeding successors $j$ is
$$
D_i(\tau)=\tau\mu_i+\left(\sum_{(i,j)\in A}\{\phi_{ij}(D_j(\tau)-\tau\mu_j)\}^p\right)^{1/p},\qquad p\ge 1.
$$
Here $p=1$ is the no-pooling sum, $p=2$ combines the safety terms as if the streams were independent, and larger $p$ represents more pooling. The planner picks $p$ to reflect how correlated the downstream demands are. That's a modeling input, not part of the optimization — the DP just consumes whatever bound function results. In the normal-demand implementation I will use for code, preprocessing pushes means upstream and adds downstream variances, so the solver sees each stage's net mean and net standard deviation and then uses the specialized safety-stock term $z\,\sigma_i^{net}\sqrt{\tau}$.

Now to code. The state is, for each relabeled node, either an $f_k$ table (keyed by outbound $S$) or a $g_k$ table (keyed by inbound $SI$). The clean recurrence can write the child terms at the boundary as $f_i(SI)$ and $g_j(S)$, but in code I still want to enumerate the allowed child service times and store the argmins; that keeps external inbound/outbound caps and the backtrack explicit. I'll mirror the implementation structure: validate demand at sink nodes, preprocess net demand and maximum replenishment times, relabel the tree, fill the two theta tables, and then backtrack committed service times.

```python
import math

def min_of_dict(values):
    arg = min(values, key=values.get)
    return values[arg], arg

def optimize_committed_service_times(tree):
    for n in tree.sink_nodes:
        if n.demand_source.mean is None:
            raise ValueError(f"Sink node {n.index} needs a demand mean.")
        if n.demand_source.standard_deviation is None:
            raise ValueError(f"Sink node {n.index} needs a demand standard deviation.")

    tree = preprocess_tree(tree)                # net demand, M_k, external CST defaults
    tree = relabel_nodes(tree)                  # each k<N has one larger adjacent node p(k)
    opt_cst_relabeled, opt_cost = _cst_dp_tree(tree)
    opt_cst = {k.original_label: opt_cst_relabeled[k.index] for k in tree.nodes}
    return opt_cst, opt_cost

def _cst_dp_tree(tree):
    theta_in = {k.index: {} for k in tree.nodes}
    theta_out = {k.index: {} for k in tree.nodes}
    best_cst_adjacent = {
        k.index: {S: {} for S in range(k.max_replenishment_time + 1)}
        for k in tree.nodes
    }
    min_k, max_k = min(tree.node_indices), max(tree.node_indices)

    for k_index in range(min_k, max_k + 1):
        k = tree.nodes_by_index[k_index]
        M, T = k.max_replenishment_time, k.processing_time
        if k_index < max_k and k.larger_adjacent_node_is_downstream:
            for S in range(M + 1):
                theta_out[k_index][S], best_cst_adjacent[k_index][S] = (
                    _calculate_theta_out(tree, k_index, S, theta_in, theta_out)
                )
            for S in range(M + 1, tree.max_max_replenishment_time + 1):
                theta_out[k_index][S] = theta_out[k_index][M]
                best_cst_adjacent[k_index][S] = best_cst_adjacent[k_index][M]
        else:
            for SI in range(M - T + 1):
                theta_in[k_index][SI], best_cst_adjacent[k_index][SI] = (
                    _calculate_theta_in(tree, k_index, SI, theta_in, theta_out)
                )
            for SI in range(M - T + 1, tree.max_max_replenishment_time + 1):
                theta_in[k_index][SI] = theta_in[k_index][M - T]
                best_cst_adjacent[k_index][SI] = best_cst_adjacent[k_index][M - T]

    final = tree.nodes_by_index[max_k]
    best_theta_in, best_SI = min_of_dict({
        SI: theta_in[max_k][SI]
        for SI in range(final.max_replenishment_time - final.processing_time + 1)
    })
    opt_cst = _backtrack_cst(tree, best_cst_adjacent, best_SI)
    return opt_cst, best_theta_in

def _calculate_theta_out(tree, k_index, S, theta_in, theta_out):
    k = tree.nodes_by_index[k_index]
    if S > k.external_outbound_cst:
        return math.inf, {}

    best = math.inf
    best_adjacent = {}
    local_S = min(S, k.external_outbound_cst)
    lo = max(k.external_inbound_cst, local_S - k.processing_time)
    hi = k.max_replenishment_time - k.processing_time
    for SI in range(lo, hi + 1):
        cost, _, best_upstream_S, best_downstream_SI = (
            _calculate_c(tree, k_index, local_S, SI, theta_in, theta_out)
        )
        if cost < best:
            best = cost
            best_adjacent = {k_index: SI}
            best_adjacent.update(best_upstream_S)
            best_adjacent.update(best_downstream_SI)
    return best, best_adjacent

def _calculate_theta_in(tree, k_index, SI, theta_in, theta_out):
    k = tree.nodes_by_index[k_index]
    best = math.inf
    best_adjacent = {}
    local_SI = max(SI, k.external_inbound_cst)
    hi = min(local_SI + k.processing_time, k.external_outbound_cst)
    for S in range(hi + 1):
        cost, _, best_upstream_S, best_downstream_SI = (
            _calculate_c(tree, k_index, S, local_SI, theta_in, theta_out)
        )
        if cost < best:
            best = cost
            best_adjacent = {k_index: S}
            best_adjacent.update(best_upstream_S)
            best_adjacent.update(best_downstream_SI)
    return best, best_adjacent

def _calculate_c(tree, k_index, S, SI, theta_in, theta_out):
    k = tree.nodes_by_index[k_index]
    tau = SI + k.processing_time - S
    safety_stock = (
        k.demand_bound_constant
        * k.net_demand_standard_deviation
        * math.sqrt(tau)
    )
    cost = k.holding_cost * safety_stock
    best_upstream_S, best_downstream_SI = {}, {}

    for i in k.predecessor_indices():
        if i < k_index:
            values = {S2: theta_out[i][S2] for S2 in range(SI + 1)}
            add_cost, best_upstream_S[i] = min_of_dict(values)
            cost += add_cost

    for j in k.successor_indices():
        if j < k_index:
            values = {
                SI2: theta_in[j][SI2]
                for SI2 in range(S, tree.max_max_replenishment_time + 1)
            }
            add_cost, best_downstream_SI[j] = min_of_dict(values)
            cost += add_cost

    return cost, k.holding_cost * safety_stock, best_upstream_S, best_downstream_SI

def _backtrack_cst(tree, best_cst_adjacent, best_SI):
    min_k, max_k = min(tree.node_indices), max(tree.node_indices)
    opt_cst, opt_in_cst = {}, {}

    for k_index in range(max_k, min_k - 1, -1):
        k = tree.nodes_by_index[k_index]
        if k_index < max_k:
            pk = k.larger_adjacent_node
            pk_is_downstream = k.larger_adjacent_node_is_downstream
            if pk < max_k:
                ppk_is_downstream = tree.nodes_by_index[pk].larger_adjacent_node_is_downstream

        if k_index == max_k:
            opt_cst[k_index] = best_cst_adjacent[k_index][best_SI][k_index]
            opt_in_cst[k_index] = best_SI
        elif pk_is_downstream:
            if pk != max_k and ppk_is_downstream:
                opt_cst[k_index] = best_cst_adjacent[pk][opt_cst[pk]][k_index]
            else:
                opt_cst[k_index] = best_cst_adjacent[pk][opt_in_cst[pk]][k_index]
            opt_in_cst[k_index] = best_cst_adjacent[k_index][opt_cst[k_index]][k_index]
        else:
            if pk != max_k and ppk_is_downstream:
                opt_in_cst[k_index] = best_cst_adjacent[pk][opt_cst[pk]][k_index]
            else:
                opt_in_cst[k_index] = best_cst_adjacent[pk][opt_in_cst[pk]][k_index]
            opt_cst[k_index] = best_cst_adjacent[k_index][opt_in_cst[k_index]][k_index]

        opt_cst[k_index] = min(opt_cst[k_index], k.external_outbound_cst)

    return opt_cst
```

The causal chain, start to finish: the single stage gives safety stock $z\,\sigma\sqrt{L}$, but in a network "how long am I exposed" is set by my neighbors, so a stage promises an outbound service time and demands an inbound one; bounding demand and handling the excess with extraordinary measures makes a 100% promise finite and the problem deterministic; the exposure window collapses to the net replenishment time $\tau = SI + T - S$, and safety stock cost is $h\,[D(\tau)-\tau\mu]$; pipeline stock is service-time-independent so it drops out; the per-stage cost is concave in the service times because $D$ is concave and $\tau$ is affine, so the optimum sits at an extreme point — Simpson's all-or-nothing falls out, the buffer concentrates; restricting to a spanning tree makes every subnetwork connect to the rest through one arc, so a DP with a single service-time state per stage works; relabel by leaves, define $f_k$ over outbound and $g_k$ over inbound service, link them by $c_k$, enumerate the finite service-time ranges while storing the child argmins needed for backtracking, and recover the placement in $O(NM^2)$.
