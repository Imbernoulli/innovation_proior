# The Guaranteed-Service Model (GSM) for strategic safety-stock placement

## Problem

A supply chain is a network of stages (raw material → component → subassembly → assembly → distribution → customer), branching into assembly (many suppliers, one stage) and distribution (one stage, many customers). End-item demand is uncertain. Decide, across the whole network at once, **where to hold safety stock and how much**, to guarantee customer service at minimum total holding cost. The difficulty is coupling: a stage that holds little safety stock leans on its supplier, so its exposure — and required stock — depends on the supplier's stock, recursively up the chain. Optimizing stages independently double-counts protection.

## Key idea

Model each stage with a periodic-review base-stock policy and two service times:

- **Outbound (guaranteed) service time** $S_j$: stage $j$ promises to fill any order within $S_j$ periods, 100% of the time.
- **Inbound service time** $SI_j$: the time $j$ waits for its inputs; $SI_j \ge S_i$ for every supplier $i$ (a supplier's outbound promise becomes the customer's inbound wait). This is the link that couples the network — deterministically.

Assume demand over $\tau$ periods is **bounded** by a known increasing concave $D(\tau)$ (e.g. $D(\tau)=\tau\mu + z\sigma\sqrt{\tau}$); demand exceeding the bound is absorbed by **extraordinary measures** (expedite, overtime, subcontract), not propagated. This makes a 100% guarantee achievable with finite stock and turns a stochastic problem into a deterministic one, with no assumption on the demand distribution.

The only exposure window not covered by a promise is the **net replenishment time**
$$\tau_j = SI_j + T_j - S_j \quad(T_j=\text{production lead time}).$$
The least base stock giving 100% service is $B_j = D_j(\tau_j)$, and expected safety stock is
$$E[I_j] = D_j(\tau_j) - \tau_j\mu_j \;\;\xrightarrow{\text{normal}}\;\; z\,\sigma_j\sqrt{\tau_j}.$$
Pipeline stock $E[W_j]=T_j\mu_j$ is independent of the service times, so it drops out of the optimization.

## The optimization and why it is tractable

$$\mathbf{P}:\ \min \sum_j h_j\big[D_j(SI_j+T_j-S_j) - (SI_j+T_j-S_j)\mu_j\big]$$
$$\text{s.t. } S_j - SI_j \le T_j,\quad SI_j - S_i \ge 0\ \forall (i,j),\quad S_j \le s_j\ (\text{demand nodes}),\quad S_j,SI_j\ge 0,\ \text{integer}.$$

Because $D_j$ is concave/nondecreasing and $\tau_j$ is affine in the service times, each per-stage cost is **concave** in $(S,SI)$; the sum is concave over a closed bounded convex region. A concave function attains its minimum at an **extreme point** — so optimal service times are extremal (Simpson's *all-or-nothing*: on a serial line, each stage has $S_i=0$ or $S_i=S_{i+1}+T_i$). The buffer concentrates at a few strategic stages.

## Spanning-tree dynamic program

On a **spanning tree**, any subnetwork connects to the rest through exactly one arc, so a DP with a *single* service-time state per stage suffices.

1. **Relabel** nodes $1..N$ so node $k<N$ has exactly one larger-labeled neighbor $p(k)$: repeatedly label a node adjacent to $\le 1$ other unlabeled node (a leaf of the remaining subgraph). $N_k$ = the subnetwork on labels $\{1..k\}$ connected to $k$.
2. **Per-stage cost** (min-convolution):
$$c_k(S,SI)=h_k\big[D_k(SI+T_k-S)-(SI+T_k-S)\mu_k\big]+\!\!\sum_{(i,k)\in A,\,i<k}\!\!f_i(SI)+\!\!\sum_{(k,j)\in A,\,j<k}\!\!g_j(S).$$
$f_i$ (min cost of $N_i$ vs. $i$'s outbound time) is **nonincreasing**, so the clean recurrence sets the supplier's outbound time to the boundary $SI$; $g_j$ (min cost of $N_j$ vs. $j$'s inbound time) is **nondecreasing**, so it sets the customer's inbound time to the boundary $S$. The implementation still enumerates the feasible child ranges $S_i\le SI$ and $SI_j\ge S$ to preserve external caps and store argmins for backtracking.
3. **Functional equations**:
$$f_k(S)=\min_{\max(0,S-T_k)\le SI\le M_k-T_k} c_k(S,SI),\qquad g_k(SI)=\min_{0\le S\le SI+T_k} c_k(S,SI),$$
with $M_k=T_k+\max\{M_i:(i,k)\in A\}$ the maximum replenishment time.
4. **Sweep** $k=1..N-1$: compute $f_k$ if $p(k)$ is downstream, else $g_k$. At $k=N$, minimize $g_N(SI)$ over $SI$ for the optimal cost; backtrack the stored argmins for every $S_j,SI_j$, hence $\tau_j$ and safety stock.

Complexity $O(NM^2)$, $M\le\sum_j T_j$ — polynomial, effectively instant for tens of stages.

## Code

The normal-bound implementation stores only the safety-stock term $z\,\sigma^{net}\sqrt{\tau}$ in the stage cost; preprocessing has already pushed end-item means and variances upstream and computed each node's maximum replenishment time.

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

    tree = preprocess_tree(tree)
    tree = relabel_nodes(tree)
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

    best, best_adjacent = math.inf, {}
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
    best, best_adjacent = math.inf, {}
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

## Notes

- **GSM vs. stochastic-service (Clark & Scarf 1960).** The stochastic-service model lets upstream stages stock out, so replenishment delays are random; it uses echelon base stocks and is exact for serial/assembly but does not collapse to a one-variable DP on mixed trees, and service times are outcomes. The GSM makes service times *decisions* via bounded demand + extraordinary measures, yielding the concave, tree-decomposable problem above; the relaxed serial problem below is the natural diagnostic for the inventory cost of requiring internal guarantees.
- **Relaxed internal guarantees.** In a serial line without guaranteed internal service, external 100% service is equivalent to $B_1+\cdots+B_i\ge D(T_1+\cdots+T_i)$ for every $i$. With nonnegative echelon holding costs, the exchange argument shifts slack from $B_k$ to $B_{k+1}$ without increasing cost, so all cumulative constraints bind: $B_1=D(T_1)$ and $B_i=D(T_1+\cdots+T_i)-D(T_1+\cdots+T_{i-1})$.
- **Multiple successors.** Combine downstream demand bounds for upstream stage $i$ via $D_i(\tau)=\tau\mu_i+\left(\sum_{(i,j)}\{\phi_{ij}(D_j(\tau)-\tau\mu_j)\}^p\right)^{1/p}$, $p\ge1$ ($p=1$ no pooling, $p=2$ independent-stream pooling); a modeling input the DP consumes.
- **Lineage.** Single-stage base stock → Simpson (1958) serial all-or-nothing → Graves (1988) serial shortest path; Inderfurth (1991, 1993), Minner (1997), Inderfurth & Minner (1998) assembly/distribution DPs → the spanning-tree DP unifying all of them.
