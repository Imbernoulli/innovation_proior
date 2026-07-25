The problem is to decide where safety stock should live in a multi-echelon supply chain and how much each stage should hold. End-customer demand is uncertain, stages branch into assembly regions where many suppliers feed one stage and distribution regions where one stage feeds many customers, and every stage's inventory requirement depends on how quickly its suppliers can replenish it. Optimizing each stage independently double-counts protection: a component that holds a large buffer makes its downstream subassembly more reliable, which lowers the subassembly's required buffer, and vice versa. The stochastic-service approach tracks echelon inventory and lets upstream stockouts create random delays; it is exact for serial or assembly systems, but the delays couple the stages through induced penalty functions, service times emerge as outcomes rather than decisions, and the machinery does not collapse into a clean single-state recursion for mixed convergent-divergent topologies. The pure guaranteed-service dynamic programs for serial, assembly, and distribution networks each handle only one topology. What is missing is a unified formulation and algorithm that treats service times as explicit decisions, works on the mixed tree-shaped networks that arise in practice, and remains fast enough to be used as a planning tool.

I propose the Guaranteed-Service Model, abbreviated GSM. The idea is to have every stage promise a guaranteed outbound service time to its customers and to honor that promise by holding enough safety stock to cover demand up to a known bound over a finite exposure window. Demand that exceeds the bound is absorbed by extraordinary measures such as expediting or overtime rather than being allowed to propagate downstream. This bounded-demand assumption turns a stochastic coupling problem into a deterministic one and makes service times explicit decision variables instead of random outcomes. For stage j with production lead time T_j, outbound service time S_j, and inbound service time SI_j, the only exposure not already covered by promises is the net replenishment time tau_j = SI_j + T_j - S_j. The least base stock that guarantees service is D_j(tau_j), where D_j is the demand bound; for the common normal-style bound D_j(tau) = tau mu_j + z sigma_j sqrt(tau), the expected safety stock is z sigma_j sqrt(tau_j). Pipeline stock T_j mu_j is independent of the service times, so it drops out of the optimization and only safety stock is optimized.

The total safety-stock holding cost is the sum over stages of h_j times the expected safety stock. Because D_j is increasing and concave and tau_j is affine in the service times, each per-stage cost is concave in (S_j, SI_j); minimizing a concave function over a closed bounded polyhedron pushes the optimum to an extreme point. That is the source of the all-or-nothing property seen in serial lines: each stage either holds enough stock to decouple completely from downstream or holds none and simply passes its inbound-plus-lead-time through as its outbound service time. On a general acyclic network with undirected cycles, a subnetwork can connect to the rest through several arcs at once, so a single service-time state would not suffice. The GSM therefore restricts the network to a spanning tree. On a tree, any connected subnetwork attaches to the remainder through exactly one arc, so the whole subnetwork can be summarized by a function of the single service time on that arc. A dynamic program with one state per stage becomes possible.

The algorithm relabels the nodes so that every node k < N has exactly one higher-labeled neighbor p(k). This is always possible on a tree by repeatedly peeling leaves. For each k we build the subnetwork N_k consisting of k and all lower-labeled pieces already attached to it. If p(k) is downstream of k, we compute f_k(S), the minimum cost of N_k as a function of k's outbound service time S. If p(k) is upstream of k, we compute g_k(SI), the minimum cost of N_k as a function of k's inbound service time SI. The per-stage cost c_k(S, SI) is k's own safety-stock cost plus the best costs of already-processed suppliers evaluated at SI and already-processed customers evaluated at S. The functional equations are f_k(S) = min over SI of c_k(S, SI) and g_k(SI) = min over S of c_k(S, SI). After sweeping k = 1 through N, the optimal cost is the minimum of g_N(SI) over SI, and backtracking the stored argmins recovers every stage's committed outbound service time S_j. Each S_j together with the committed times of its suppliers pins SI_j, hence tau_j and the expected safety stock z sigma_j sqrt(tau_j). Each minimization is over a finite integer range, so the complexity is O(N M^2) with M bounded by the sum of lead times; this is polynomial and easily fast enough for tens of stages.

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
