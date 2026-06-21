The problem is to decide where safety stock should live in a multi-echelon supply chain and how much each stage should hold. End-customer demand is uncertain, stages branch into assembly regions where many suppliers feed one stage and distribution regions where one stage feeds many customers, and every stage's inventory requirement depends on how quickly its suppliers can replenish it. Optimizing each stage independently double-counts protection: a component that holds a large buffer makes its downstream subassembly more reliable, which lowers the subassembly's required buffer, and vice versa. The stochastic-service approach tracks echelon inventory and lets upstream stockouts create random delays; it is exact for serial or assembly systems, but the delays couple the stages through induced penalty functions, service times emerge as outcomes rather than decisions, and the machinery does not collapse into a clean single-state recursion for mixed convergent-divergent topologies. The pure guaranteed-service dynamic programs for serial, assembly, and distribution networks each handle only one topology. What is missing is a unified formulation and algorithm that treats service times as explicit decisions, works on the mixed tree-shaped networks that arise in practice, and remains fast enough to be used as a planning tool.

I propose the Guaranteed-Service Model, abbreviated GSM. The idea is to have every stage promise a guaranteed outbound service time to its customers and to honor that promise by holding enough safety stock to cover demand up to a known bound over a finite exposure window. Demand that exceeds the bound is absorbed by extraordinary measures such as expediting or overtime rather than being allowed to propagate downstream. This bounded-demand assumption turns a stochastic coupling problem into a deterministic one and makes service times explicit decision variables instead of random outcomes. For stage j with production lead time T_j, outbound service time S_j, and inbound service time SI_j, the only exposure not already covered by promises is the net replenishment time tau_j = SI_j + T_j - S_j. The least base stock that guarantees service is D_j(tau_j), where D_j is the demand bound; for the common normal-style bound D_j(tau) = tau mu_j + z sigma_j sqrt(tau), the expected safety stock is z sigma_j sqrt(tau_j). Pipeline stock T_j mu_j is independent of the service times, so it drops out of the optimization and only safety stock is optimized.

The total safety-stock holding cost is the sum over stages of h_j times the expected safety stock. Because D_j is increasing and concave and tau_j is affine in the service times, each per-stage cost is concave in (S_j, SI_j); minimizing a concave function over a closed bounded polyhedron pushes the optimum to an extreme point. That is the source of the all-or-nothing property seen in serial lines: each stage either holds enough stock to decouple completely from downstream or holds none and simply passes its inbound-plus-lead-time through as its outbound service time. On a general acyclic network with undirected cycles, a subnetwork can connect to the rest through several arcs at once, so a single service-time state would not suffice. The GSM therefore restricts the network to a spanning tree. On a tree, any connected subnetwork attaches to the remainder through exactly one arc, so the whole subnetwork can be summarized by a function of the single service time on that arc. A dynamic program with one state per stage becomes possible.

The algorithm relabels the nodes so that every node k < N has exactly one higher-labeled neighbor p(k). This is always possible on a tree by repeatedly peeling leaves. For each k we build the subnetwork N_k consisting of k and all lower-labeled pieces already attached to it. If p(k) is downstream of k, we compute f_k(S), the minimum cost of N_k as a function of k's outbound service time S. If p(k) is upstream of k, we compute g_k(SI), the minimum cost of N_k as a function of k's inbound service time SI. The per-stage cost c_k(S, SI) is k's own safety-stock cost plus the best costs of already-processed suppliers evaluated at SI and already-processed customers evaluated at S. The functional equations are f_k(S) = min over SI of c_k(S, SI) and g_k(SI) = min over S of c_k(S, SI). After sweeping k = 1 through N, the optimal cost is the minimum of g_N(SI) over SI, and backtracking the stored argmins recovers every S_j, SI_j, tau_j, and safety stock. Each minimization is over a finite integer range, so the complexity is O(N M^2) with M bounded by the sum of lead times; this is polynomial and easily fast enough for tens of stages.

```python
import math
from collections import defaultdict


def guaranteed_service_placement(
    nodes,
    arcs,
    lead_time,
    holding_cost,
    demand_mean,
    demand_std,
    z,
    max_service_time=None,
    phi=None,
):
    """
    nodes: list of node identifiers
    arcs: list of (i, j) directed arcs meaning i supplies j
    lead_time: dict node -> deterministic production lead time T_j
    holding_cost: dict node -> per-unit holding cost h_j
    demand_mean: dict node -> end-item mean demand (nonzero only at sinks)
    demand_std: dict node -> end-item standard deviation (nonzero only at sinks)
    z: service coefficient for normal-style demand bound D(tau) = tau*mu + z*sigma*sqrt(tau)
    max_service_time: dict node -> external outbound cap at demand nodes (default 0 at sinks)
    phi: dict arc -> units of upstream item per downstream unit (default 1.0)
    Returns: dict node -> outbound service time S_j, dict node -> safety stock, optimal cost.
    """
    if phi is None:
        phi = {(i, j): 1.0 for (i, j) in arcs}
    if max_service_time is None:
        max_service_time = {j: (0 if not any(j == i for (i, _) in arcs) else float('inf'))
                            for j in nodes}

    succ = defaultdict(list)
    pred = defaultdict(list)
    for (i, j) in arcs:
        succ[i].append(j)
        pred[j].append(i)

    # Push end-item demand upstream so every stage sees its net mean and std.
    net_mu = {j: 0.0 for j in nodes}
    net_sigma = {j: 0.0 for j in nodes}
    sinks = [j for j in nodes if j not in succ or not succ[j]]
    for j in sinks:
        net_mu[j] = demand_mean.get(j, 0.0)
        net_sigma[j] = demand_std.get(j, 0.0)

    visited = set()
    order = []

    def post(v):
        visited.add(v)
        for u in pred[v]:
            if u not in visited:
                post(u)
        order.append(v)

    for s in sinks:
        if s not in visited:
            post(s)

    for j in reversed(order):
        for i in pred[j]:
            net_mu[i] += phi[(i, j)] * net_mu[j]
            net_sigma[i] = math.hypot(net_sigma[i], phi[(i, j)] * net_sigma[j])

    # Maximum replenishment time M_j: longest path from j down to a sink.
    M = {}
    for j in reversed(order):
        if not succ[j]:
            M[j] = lead_time[j]
        else:
            M[j] = lead_time[j] + max(M[k] for k in succ[j])

    # Relabel nodes so each k < N has exactly one higher-labeled neighbor p(k).
    adj = {j: set(pred[j]) | set(succ[j]) for j in nodes}
    unlabeled = set(nodes)
    label = {}
    reverse = {}
    current = 1
    while unlabeled:
        leaf = next(v for v in unlabeled if len(adj[v] & unlabeled) <= 1)
        label[leaf] = current
        reverse[current] = leaf
        unlabeled.remove(leaf)
        current += 1

    N = len(nodes)

    # For each label k, collect already-processed suppliers/customers and the one higher neighbor.
    suppliers_done = {k: [] for k in range(1, N + 1)}
    customers_done = {k: [] for k in range(1, N + 1)}
    p = {}
    downstream = {}
    for k in range(1, N + 1):
        j = reverse[k]
        higher = []
        for i in pred[j]:
            li = label[i]
            if li < k:
                suppliers_done[k].append(li)
            elif li > k:
                higher.append(li)
        for c in succ[j]:
            lc = label[c]
            if lc < k:
                customers_done[k].append(lc)
            elif lc > k:
                higher.append(lc)
        if k < N:
            pk = max(higher)
            p[k] = pk
            downstream[k] = pk in [label[c] for c in succ[j]]

    # Bound helpers for service-time ranges.
    def max_inbound(j):
        if not pred[j]:
            return 0
        return max(M[i] for i in pred[j])

    def max_outbound(j):
        return M[j]

    def safety_stock_cost(j, tau):
        tau = max(0, tau)
        return holding_cost[j] * z * net_sigma[j] * math.sqrt(tau)

    theta_out = {k: {} for k in range(1, N + 1)}
    theta_in = {k: {} for k in range(1, N + 1)}
    best_adj = {k: {} for k in range(1, N + 1)}

    def get_theta_out(i, SI):
        table = theta_out[i]
        if not table:
            return 0.0
        key = min(SI, max(table.keys()))
        return table.get(key, float('inf'))

    def get_theta_in(c, S):
        table = theta_in[c]
        if not table:
            return 0.0
        key = min(S, max(table.keys()))
        return table.get(key, float('inf'))

    for k in range(1, N + 1):
        j = reverse[k]
        T = lead_time[j]
        cap = int(max_service_time[j]) if max_service_time[j] != float('inf') else M[j]
        if k < N:
            pk = p[k]
            if downstream[k]:
                # f_k(S): minimize over inbound SI.
                for S in range(min(M[j], cap) + 1):
                    best_val = float('inf')
                    best_SI = None
                    lo = max(0, S - T)
                    hi = max_inbound(j)
                    for SI in range(lo, hi + 1):
                        tau = SI + T - S
                        val = safety_stock_cost(j, tau)
                        for i in suppliers_done[k]:
                            val += get_theta_out(i, SI)
                        for c in customers_done[k]:
                            val += get_theta_in(c, S)
                        if val < best_val:
                            best_val = val
                            best_SI = SI
                    theta_out[k][S] = best_val
                    best_adj[k][S] = best_SI
            else:
                # g_k(SI): minimize over outbound S.
                for SI in range(max_inbound(j) + 1):
                    best_val = float('inf')
                    best_S = None
                    hi = min(SI + T, cap, M[j])
                    for S in range(hi + 1):
                        tau = SI + T - S
                        val = safety_stock_cost(j, tau)
                        for i in suppliers_done[k]:
                            val += get_theta_out(i, SI)
                        for c in customers_done[k]:
                            val += get_theta_in(c, S)
                        if val < best_val:
                            best_val = val
                            best_S = S
                    theta_in[k][SI] = best_val
                    best_adj[k][SI] = best_S
        else:
            # Root: g_N(SI).
            for SI in range(max_inbound(j) + 1):
                best_val = float('inf')
                best_S = None
                hi = min(SI + T, cap, M[j])
                for S in range(hi + 1):
                    tau = SI + T - S
                    val = safety_stock_cost(j, tau)
                    for i in suppliers_done[k]:
                        val += get_theta_out(i, SI)
                    for c in customers_done[k]:
                        val += get_theta_in(c, S)
                    if val < best_val:
                        best_val = val
                        best_S = S
                theta_in[k][SI] = best_val
                best_adj[k][SI] = best_S

    # Backtrack from the root.
    root = N
    best_SI, best_cost = min(theta_in[root].items(), key=lambda x: x[1])
    S_opt = {}
    SI_opt = {}
    SI_opt[root] = best_SI
    S_opt[root] = best_adj[root][best_SI]

    for k in range(N - 1, 0, -1):
        pk = p[k]
        if downstream[k]:
            # k supplies p(k), so S_k equals parent's inbound SI.
            S_opt[k] = SI_opt[pk]
            SI_opt[k] = best_adj[k][S_opt[k]]
        else:
            # p(k) supplies k, so SI_k equals parent's outbound S.
            SI_opt[k] = S_opt[pk]
            S_opt[k] = best_adj[k][SI_opt[k]]

    # Map labels back to original nodes and compute safety stocks.
    S_node = {reverse[k]: S_opt[k] for k in range(1, N + 1)}
    SI_node = {reverse[k]: SI_opt[k] for k in range(1, N + 1)}
    safety = {}
    for j in nodes:
        tau = SI_node[j] + lead_time[j] - S_node[j]
        safety[j] = z * net_sigma[j] * math.sqrt(max(0, tau))

    return S_node, safety, best_cost


if __name__ == "__main__":
    # Simple three-stage serial line for validation.
    nodes = [1, 2, 3]
    arcs = [(1, 2), (2, 3)]
    lead_time = {1: 4, 2: 4, 3: 4}
    holding_cost = {1: 1.0, 2: 0.5, 3: 0.2}
    demand_mean = {1: 0, 2: 0, 3: 10.0}
    demand_std = {1: 0, 2: 0, 3: 3.0}
    z = 2.0
    max_service_time = {1: float('inf'), 2: float('inf'), 3: 0}

    S, safety, cost = guaranteed_service_placement(
        nodes, arcs, lead_time, holding_cost,
        demand_mean, demand_std, z, max_service_time
    )
    print("Outbound service times:", S)
    print("Safety stocks:", safety)
    print("Optimal cost:", cost)
```
