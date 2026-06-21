The problem is to determine the effect of an intervention from passive observational data. We see how variables co-vary when nature generates them, but we want to know what would happen if we fixed one variable to a chosen value and let the rest of the system continue as before. Standard approaches such as regression, matching, or stratification try to repair the comparison by conditioning on covariates, but conditioning is not the same as intervening. Conditioning selects units whose value arose naturally from the system, while setting a value replaces the mechanism that produced it. Without a clear account of which paths are opened or closed by conditioning, these methods can silently introduce bias through colliders, fail to block back-door confounding, or break down entirely when the causal route runs through a mediator affected by treatment.

What is needed is an identification criterion: a way to decide whether the combination of observed data and explicit causal assumptions is enough to determine the interventional quantity, and to expose cases where two causal models agree on all observations yet disagree on the intervention effect. The right language turns out to be a calculus of graph transformations rather than a catalog of adjustment tricks.

The method is do-calculus. It treats a causal diagram as a compact encoding of structural mechanisms, where each node's parents are the inputs to its local generating equation. An intervention do(x) replaces the equation for X with the constant x, which corresponds graphically to deleting all arrows into X. The probability expression and the graph surgery move together: whenever the target contains do(x), the relevant independence questions are asked in the graph where X's incoming arrows have been removed.

Do-calculus provides three rewrite rules over mixed expressions containing both observations and interventions. The first rule allows an observation to be inserted or deleted: P(y | do(x), z, w) equals P(y | do(x), w) when Y is d-separated from Z given X and W in the graph with arrows into X removed. The second rule exchanges an action for an observation: P(y | do(x), do(z), w) equals P(y | do(x), z, w) when Y is d-separated from Z given X and W in the graph where arrows into X and arrows out of Z have been removed. The third rule deletes an action: P(y | do(x), do(z), w) equals P(y | do(x), w) when Y is d-separated from Z given X and W in the graph where arrows into X are removed and incoming arrows are deleted only for those members of Z that are not ancestors of W in that already mutilated graph. Between rule applications, ordinary probability manipulations such as marginalization and conditioning are used freely.

A causal effect is identified when repeated application of these rules removes every do(...) symbol from the target expression, leaving only observational probabilities. The familiar back-door adjustment is one special case. If a set Z blocks every back-door path from treatment X to outcome Y and contains no descendant of X, then do-calculus reduces P(y | do(x)) to sum_z P(y | x, z)P(z). The front-door adjustment is the classic illustration of why the calculus is stronger than informal control selection. When a mediator Z intercepts all directed paths from X to Y, there is no unblocked back-door path from X to Z, and all back-door paths from Z to Y are blocked by X, then do-calculus derives P(y | do(x)) = sum_z P(z | x) sum_{x'} P(y | z, x')P(x'), even though X and Y share an unobserved common cause. The target is decomposed into pieces, and different transformed graphs license the required exchanges at each step.

The method is also a warning system. If a derivation leaves an irreducible do(...) term, the graph structure implies that two structural models can produce identical observed probabilities while differing on the intervention effect. In that case no amount of clever regression can recover the effect from the data under the stated assumptions. Later completeness results confirm that, for causal Bayesian networks, the three rules together with ordinary probability manipulation are sufficient to derive every identifiable causal effect.

```python
from functools import lru_cache
from itertools import combinations


def parents(node, edges):
    """Return the set of parents of node in a directed edge list."""
    return {u for u, v in edges if v == node}


def ancestors(nodes, edges):
    """Return the set of ancestors of the given nodes."""
    result = set(nodes)
    changed = True
    while changed:
        changed = False
        for u, v in edges:
            if v in result and u not in result:
                result.add(u)
                changed = True
    return result


def d_separated(x, y, z, edges, bidirected=None):
    """
    Simple d-separation test for a graph with directed edges and optional
    bidirected confounding edges.  Nodes are strings; z is a conditioning set.
    """
    if bidirected is None:
        bidirected = set()
    # Moralize the ancestral graph: start with ancestors of X, Y, Z.
    relevant = ancestors({x, y} | set(z), edges | {(b, a) for a, b in bidirected})
    # Build moral edges: connect parents of each node.
    moral = set()
    for node in relevant:
        pars = [p for p in parents(node, edges) if p in relevant]
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                moral.add(tuple(sorted((pars[i], pars[j]))))
    # Treat bidirected edges as undirected.
    undirected = set()
    for u, v in edges:
        if u in relevant and v in relevant:
            undirected.add(tuple(sorted((u, v))))
    for a, b in bidirected:
        if a in relevant and b in relevant:
            undirected.add(tuple(sorted((a, b))))
    undirected |= moral

    # Bayes-ball style traversal through the mixed graph.
    visited = set()
    # state is (current_node, came_from_parent)
    start_state = (x, False)
    stack = [start_state]
    while stack:
        node, from_parent = stack.pop()
        if (node, from_parent) in visited:
            continue
        visited.add((node, from_parent))
        if node == y:
            return False
        if node in z:
            # Conditioning blocks trails coming from a child but allows trails
            # from a parent to continue (collider activation handled below).
            if from_parent:
                for neighbor in neighbors_of(node, undirected):
                    stack.append((neighbor, False))
        else:
            # Continue to all neighbors; mark whether we arrived from a parent.
            for neighbor in neighbors_of(node, undirected):
                # Determine whether neighbor is a parent of node in the
                # directed edge sense (so node is a child of neighbor).
                is_parent = (neighbor, node) in edges
                stack.append((neighbor, is_parent))
    return True


def neighbors_of(node, undirected):
    """Return neighbors of node in an undirected edge set."""
    result = set()
    for u, v in undirected:
        if u == node:
            result.add(v)
        elif v == node:
            result.add(u)
    return result


def mutilate(edges, remove_into=None, remove_out_of=None):
    """Return a new edge set with specified incoming/outgoing arrows removed."""
    remove_into = set(remove_into or [])
    remove_out_of = set(remove_out_of or [])
    return {
        (u, v)
        for u, v in edges
        if not (v in remove_into or u in remove_out_of)
    }


def z_w_subset(z, w, x, edges):
    """
    Compute Z(W) for Rule 3: members of Z that are not ancestors of W in
    G_bar_X (the graph with arrows into X removed).
    """
    g_bar_x = mutilate(edges, remove_into={x})
    w_ancestors = ancestors(set(w), g_bar_x)
    return {node for node in z if node not in w_ancestors}


def rule1_licensed(y, x, z, w, edges):
    """Rule 1: insert/delete observation z in P(y | do(x), z, w)."""
    g_bar_x = mutilate(edges, remove_into={x})
    return d_separated(y, z, set(w) | {x}, g_bar_x)


def rule2_licensed(y, x, z, w, edges):
    """Rule 2: exchange do(z) for z in P(y | do(x), do(z), w)."""
    g = mutilate(edges, remove_into={x}, remove_out_of=set(z))
    return d_separated(y, frozenset(z), set(w) | {x}, g)


def rule3_licensed(y, x, z, w, edges):
    """Rule 3: insert/delete action do(z) in P(y | do(x), do(z), w)."""
    z_active = z_w_subset(set(z), set(w), x, edges)
    g = mutilate(edges, remove_into={x} | z_active)
    return d_separated(y, frozenset(z), set(w) | {x}, g)


# Example: front-door graph
# X -> Z -> Y, with a latent confounder U between X and Y.
edges = {("X", "Z"), ("Z", "Y")}
confounding = {("X", "Y")}

# Test whether do-calculus licenses the front-door adjustment.
# We want P(y | do(x)).  Decompose via marginalization over Z:
# P(y | do(x)) = sum_z P(y | do(x), z) P(z | do(x))
# Step 1: P(z | do(x)) = P(z | x) by Rule 2 (X has no parents, graph same).
licensed_step1 = rule2_licensed("Z", "X", set(), [], edges)
# Step 2: P(y | do(x), z) = sum_x' P(y | z, x') P(x') by Rule 2.
licensed_step2 = rule2_licensed("Y", "Z", {"X"}, [], edges)

print("Front-door step 1 licensed:", licensed_step1)
print("Front-door step 2 licensed:", licensed_step2)

# Example: back-door graph with a single confounder Z.
# Z -> X, Z -> Y, X -> Y
edges_backdoor = {("Z", "X"), ("Z", "Y"), ("X", "Y")}
# P(y | do(x)) = sum_z P(y | x, z) P(z) via Rule 2 on X conditioned on Z.
licensed_backdoor = rule2_licensed("Y", "X", set(), ["Z"], edges_backdoor)
print("Back-door step licensed:", licensed_backdoor)
```
