# Synthesis — Guaranteed-Service Model (GSM) for strategic safety-stock placement

## Primary source actually read
- Graves & Willems 2000, "Optimizing Strategic Safety Stock Placement in Supply Chains," MSOM 2(1):68–83. Full text incl. appendix retrieved as `refs/gw2000_uspto.pdf` (USPTO-hosted copy, KINAXIS exhibit). Read pages 1–16 in full.
- Code: stockpyl `gsm_tree.py`, `gsm_serial.py`, `gsm_helpers.py` (Larry Snyder, MIT license) vendored in `code/`. These implement the exact DP. FOSCT = Snyder & Shen "Fundamentals of Supply Chain Theory" notation (theta^o = f, theta^i = g).
- Background/explainers: Achkar et al. (EJOR 2023 / arXiv 2306.10961) extensions paper (`refs/achkar_*`), which restates the GSM/SSM contrast, net-lead-time, safety stock formula.

## The exact objects (verified against primary text)

### Single-stage base-stock (ancestor recap)
- Periodic review, base-stock policy. Order-up-to level B. Each period order back up to B; with no order delay external demand propagates up the chain so every stage sees end-customer demand.
- Newsvendor optimal base stock: critical ratio b/(b+h); B = μ_L + z·σ_L for normal lead-time demand, z = Φ^{-1}(b/(b+h)).
- Safety stock SS = z·σ·√L (fixed lead time L). z = k = z_α = "demand bound constant".

### GW2000 demand bound (eq numbering theirs)
- Each stage j: deterministic production lead time T_j. Arc multiplier φ_ij (units of i per unit of j). Internal demand d_j(t) = Σ_{(i,j)∈A} φ_ij d_j(t); μ_i = Σ φ_ij μ_j.
- Bounded demand: D_j(τ) is a bound on demand over τ periods, D_j(0)=0, increasing & **concave** in τ. Example (eq 1): D_j(τ) = τμ + k σ √τ. Eq (2) gives the risk-pooling combination for multiple successors: D_j(τ)=τμ_i + p·√(Σ {φ_ij(D_j(τ)' − τμ_j)}^p), p≥1; p=1 no pooling, p=2 = combine std devs of independent streams.
- Excess demand beyond the bound handled by **extraordinary measures** (expedite, overtime, subcontract) — *operating flexibility*. No assumption on demand distribution.
- M_j = max replenishment time = T_j + max{M_i : (i,j)∈A} (footnote 1).

### Service times
- Outbound guaranteed service time S_j: demand d_j(t) filled by t+S_j, 100% service.
- Inbound service time SI_j: time to get inputs from suppliers and begin production. SI_j ≥ max{S_i : (i,j)∈A} (eq, single supplier service time; convention SI_j = max{S_j − T_j, max S_i}, but for the optimization SI_j − S_i ≥ 0 constraint eq (8)).
- Initial dev: S_ij = S_i (same service to all customers). Customer-specific via zero-cost dummy nodes (Graves & Willems 1998).

### Single-stage inventory (§3, eqs 3–5)
- I_j(t) = B_j − d_j(t − SI_j − T_j, t − S_j). (eq 3)
- Demand bounded ⇒ 100% service with least inventory by B_j = D_j(τ), τ = SI_j + T_j − S_j. (eq 4) **net replenishment time τ = SI + T − S.**
- Expected safety stock E[I_j] = D_j(SI_j+T_j−S_j) − (SI_j+T_j−S_j)μ_j. (eq 5). With normal bound: E[I_j] = k σ √(SI+T−S) = z·σ·√τ.
- Pipeline/WIP W_j(t) = d_j(t−SI−T, t−SI), E[W_j] = T_j μ_j — independent of service times, so dropped from optimization (it's predetermined). Optimize **safety stock only**.

### Optimization problem P (§4)
min Σ_j h_j { D_j(SI_j+T_j−S_j) − (SI_j+T_j−S_j)μ_j }
s.t. S_j − SI_j ≤ T_j (i.e. net repl time τ = SI+T−S ≥ 0) for all j
     SI_j − S_i ≥ 0 for all (i,j)∈A
     S_j ≤ s_j for demand nodes (max service time)
     S_j, SI_j ≥ 0 and integer.
- Objective is **concave** in (S, SI) because D_j is concave and the net replenishment time SI+T−S is an affine function of the service times; composition of a concave nondecreasing function with an affine map is concave; sum of concave is concave. Feasible region is a closed bounded convex set (bounded because optimal service times need not exceed Σ production lead-times, as D nondecreasing). 
- **Minimizing a concave function over a closed bounded convex polytope ⇒ optimum at an extreme point** (Luenberger 1973). 
- Simpson 1958: serial line, external service time 0 ⇒ **all-or-nothing**: at optimum each stage either S_i = 0 (holds enough safety stock to fully decouple from downstream) or S_i = S_{i+1}+T_i (holds NO safety stock, just passes its inbound service time + own lead time through). Gallego & Zipkin 1999: near-optimal under traditional (non-bounded) assumptions.
- Graves (1988): serial-line problem = shortest path. Inderfurth (1991,1993), Inderfurth & Minner (1998), Minner (1997): DP for assembly / distribution networks. Graves & Willems (1996) similar for assembly/distribution. THIS paper: **spanning tree** (the general single-arc-between-any-two-subnetworks case unifying serial/assembly/distribution).

### Spanning-tree DP (§5) — the heart
**Labeling**: relabel nodes 1..N so that node k (k<N) is adjacent to exactly ONE node with higher label, denoted p(k). Algorithm: repeatedly pick an unlabeled node adjacent to ≤1 other unlabeled node, give it next label. Always possible on a tree (a tree always has a leaf in the remaining subgraph). N_k = subnetwork = {k} ∪ (upstream Σ N_i, i<k,(i,k)∈A) ∪ (downstream Σ N_j, j<k,(k,j)∈A) — the connected subgraph on labels {1..k} containing k.

**Per-stage cost** (eq for c_k):
c_k(S, SI) = h_k{ D_k(SI + T_k − S) − (SI + T_k − S)μ_k } + Σ_{(i,k)∈A, i<k} f_i(SI) + Σ_{(k,j)∈A, j<k} g_j(S).
- First term: safety stock cost at k.
- f_i(SI): min holding cost for subnetwork N_i (upstream child i) given i's outbound service = SI; f_i is **nonincreasing** in SI (more slack upstream ⇒ cheaper), so set i's outbound service = SI (the inbound service of k), no loss.
- g_j(S): min holding cost for subnetwork N_j (downstream child j) given j's inbound service = S; g_j **nondecreasing** in S, so set j's inbound service = S (the outbound service of k).

**Functional equations**:
f_k(S) = min_{SI} c_k(S,SI) s.t. max(0, S−T_k) ≤ SI ≤ M_k − T_k, SI integer. [used when p(k) downstream]
g_k(SI) = min_{S} c_k(S,SI) s.t. 0 ≤ S ≤ SI + T_k (and S ≤ s_k if demand node). [used when p(k) upstream]

**DP**: for k=1..N−1: if p(k) downstream evaluate f_k(S), S=0..M_k; if p(k) upstream evaluate g_k(SI), SI=0..M_k−T_k. For k=N evaluate g_N(SI) for all SI, then minimize over SI = optimal cost. Backtrack for optimal service times.
**Complexity O(N M^2)**, M = max service time ≤ Σ T_j. Single state variable (S or SI) per stage because tree ⇒ exactly one connecting arc to the rest.

Code mapping (stockpyl): theta_out = f, theta_in = g, _calculate_c = c_k. safety_stock = demand_bound_constant·net_demand_std·sqrt(SI+T−S). Children min over allowed range: upstream child theta_out over S2 ≤ SI; downstream child theta_in over SI2 ≥ S — exactly the f/g monotonicity selection.

### Appendix (relaxation of guaranteed internal service)
- Serial chain, relax: internal stages need NOT guarantee service to internal customers; service levels follow from base stocks chosen to minimize total holding. Stage 1 = 100% to external customer, external service time 0.
- Net inventory I_i(t)=[B_i − d(t−T_i,t) − Q_{i+1}(t−T_i)]^+, backlog Q_i(t)=[d(t−T_i,t)+Q_{i+1}(t−T_i) − B_i]^+.
- By induction Q_i(t) = max over partial sums; 100% external service ⇒ B_1+...+B_i ≥ D(T_1+...+T_i) (eq A3).
- Echelon holding cost e_i = h_i − h_{i+1}. Optimization P*: min Σ h_i B_i − Σ_{i≥2} e_{i-1} E[Q_i] s.t. eq A3, B_i ≥ 0.
- **Result**: if echelon holding costs nonnegative and D nondecreasing, optimal solution has ALL constraints (A3) binding: B_1 = D(T_1), B_i = D(T_1+..+T_i) − D(T_1+..+T_{i-1}). Proof by exchange argument (push a binding-violation Δ down the chain, objective no worse, uses E[Q] monotonicity from A2 + nonneg echelon costs). Generalizes to assembly via Rosling 1989 transformation.
- Empirical (36 test problems, Poisson demand): guaranteed-internal-service safety stock cost on avg **26% higher** (range 7–43%); total inventory cost on avg **4% higher** (range <1–14%). So the guaranteed-service simplification costs little and is "already ingrained in practice."

## GSM vs SSM (Clark & Scarf 1960 stochastic-service)
- **SSM (Clark & Scarf 1960)**: stochastic service — upstream stage may stock out, so the time a downstream stage waits for replenishment is a *random variable*; uses **echelon inventory / echelon base-stock**, decomposes serial system exactly into single-stage problems with induced penalty cost functions. No bounded-demand assumption; models stockout backorder propagation. Optimal for serial (and via Rosling, assembly), but hard for general networks; service times are outcomes, not decisions.
- **GSM (Simpson 1958 / GW2000)**: guaranteed service — every stage promises a deterministic service time and ALWAYS meets it (100%) by holding enough safety stock for *bounded* demand; excess handled by extraordinary measures. Service times are **decision variables**. Deterministic optimization, concave ⇒ extreme-point ⇒ tree DP. Tractable for richer topologies. Tradeoff: assumes bounded demand + operating flexibility; ignores within-bound stockout propagation.

## Design-decision → why table
| Decision | Why this, not the alternative |
|---|---|
| Bounded demand D(τ), excess via extraordinary measures | Makes 100% service achievable with finite stock; turns a stochastic problem into a deterministic one. SSM alternative (model backorder propagation) is exact but couples stages through random delays and is intractable on trees. Managers prefer "100% for a covered range" to estimating shortage costs. |
| Guaranteed (committed) service times as decision variables | Decouples each stage's local problem: given (SI, S), a stage's safety stock depends only on τ=SI+T−S. Lets the network problem be a clean optimization over service-time labels. |
| Net replenishment time τ = SI + T − S | The only window of demand exposure not already covered by a promise — base stock must cover exactly D(τ). Captures the coupling: a stage's S becomes its customer's SI. |
| Optimize safety stock only (drop pipeline) | E[W]=Tμ is independent of service times ⇒ constant ⇒ irrelevant to the argmin. |
| Concavity of per-stage cost | D concave + τ affine in (S,SI) ⇒ concave; sum concave. This is what forces optima to extreme points (Simpson all-or-nothing) and lets the DP enumerate only integer service-time labels rather than a continuum. |
| Integer service times | Periodic review ⇒ service times are whole periods; finite label set 0..M makes enumeration in the DP exact and finite. |
| Spanning-tree restriction + node relabeling | A tree has exactly one arc between any subnetwork N_k and the rest ⇒ a SINGLE DP state variable (S or SI) per stage. General networks would need multiple coupling variables (NP-hard in general). The leaf-labeling guarantees each k connects upward via one node p(k). |
| f nonincreasing in SI / g nondecreasing in S | Lets the parent set the child's free service time equal to the parent's coupling time without loss of optimality — collapses the min-convolution to a single argmin per child. |
| Min-convolution form of c_k | DP additivity: subnetwork cost = own safety stock + best child subnetwork costs consistent with the coupling service time. |

## Evaluation settings (pre-method yardstick, no GSM outcomes)
- Spanning-tree supply chains (serial/assembly/distribution as special cases). Stages with deterministic lead times T_j, holding cost h_j, demand mean/std at demand nodes, max service times s_j at demand nodes.
- Natural yardsticks at the time: Clark-Scarf echelon-stock SSM (serial/assembly), prior GSM DPs for serial (Graves 1988 shortest path; Inderfurth 1991), assembly/distribution (Inderfurth 1993; Graves & Willems 1996). Metric: total safety-stock holding cost (and total inventory cost incl. pipeline).
- Demand processes used in the appendix study: Poisson with mean λ∈{10,50}, truncation percentile α∈{0.90,0.98}; lead-time triples (T1,T2,T3)∈{(4,4,4),(1,3,8),(8,3,1)}; holding triples (1,0.5,0.2),(1,0.66,0.33),(1,0.8,0.5).
