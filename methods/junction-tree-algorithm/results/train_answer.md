I will now present the junction tree algorithm as the canonical method for exact probabilistic inference in graphical models. The problem begins with a collection of local factors, either from a Bayesian network encoded as conditional probability tables or from an undirected graphical model encoded as potentials. The goal is to compute posterior marginals or other queries over the joint distribution without ever materializing the full joint table. Direct summation over all unobserved variables is exponential in the number of variables, so an exact method must exploit structure. The junction tree algorithm does this by re-expressing the model as a tree whose nodes are clusters of variables large enough to capture every dependence created during exact elimination.

The first step, if the model is directed, is moralization. I connect every pair of parents that share a common child, and then I drop all edge directions. The resulting undirected graph is called the moral graph. This step is necessary because the co-parents in a directed model become coupled once their common child is observed or summed out, and moralization makes those couplings explicit as graph edges. After moralization, every factor in the model corresponds to a clique or subset of a clique in the moral graph.

The second step is triangulation. I add fill-in edges to the moral graph until every cycle of length at least four has a chord. The resulting graph is chordal, meaning it has no chordless cycles of length greater than three. Chordality is the key graph-theoretic property that makes exact inference tractable in a local message-passing framework, because it guarantees that the maximal cliques of the graph can be arranged into a tree with the running-intersection property. Without triangulation, variable elimination would still be exact, but the intermediate factors would not have a reusable tree structure. By making the fill-in explicit, I turn dynamic programming into a persistent data structure that can be calibrated once and then reused for many queries.

Once the graph is chordal, I extract its maximal cliques. Each maximal clique will become a node in the junction tree. I then connect these cliques into a tree such that, for every variable X in the original model, the set of cliques containing X forms a connected subtree. This is the running-intersection property. A standard way to construct such a tree is to form a complete graph over the maximal cliques, assign each edge a weight equal to the size of the intersection of the two cliques, and then compute a maximum-weight spanning tree. The edges of this tree are labeled by the separator sets, which are the intersections of adjacent cliques.

Next I assign each original factor to a clique that contains its entire scope, and I multiply all factors assigned to the same clique into a single clique potential. If a factor is fully contained in more than one clique, it only needs to be assigned once; otherwise the model would be double-counted. Evidence is handled locally. For hard evidence, I restrict every clique potential that contains the observed variable to the rows consistent with the observed value. For soft or virtual evidence, I multiply an additional local potential into any clique that contains the relevant variables. The global normalizing constant is left implicit until the end, when it can be recovered by summing any calibrated clique potential.

Calibration is performed by message passing over separators. Each message from clique C_i to clique C_j is a function defined on the separator S_ij = C_i intersect C_j. I compute the outgoing message by marginalizing the current belief at C_i over the variables that are not in S_ij. The receiving clique multiplies this message in and divides by any previously stored separator belief to avoid double-counting. After a full collect-and-distribute schedule, or any schedule that sends messages along every tree edge in both directions, adjacent cliques agree on their separator marginals. At that point the clique beliefs are not merely local approximations; they are the exact marginals of a single consistent joint distribution.

The reason exactness holds is the combination of chordality and running intersection. Chordality ensures that every factor created during variable elimination can be stored inside some clique. Running intersection ensures that every variable is represented consistently across the tree: if a variable appears in two cliques, it also appears in every clique on the path between them, so there is a unique way to reconcile beliefs about that variable. Once the tree is calibrated, I can read off any single-variable marginal by marginalizing any clique that contains that variable, and I can compute arbitrary marginals over connected subsets by inspecting the appropriate cliques and separators.

The computational cost of the junction tree algorithm is dominated by the size of the largest clique. For discrete variables with tabular potentials, both storage and arithmetic are exponential in the number of variables in the largest clique, or equivalently in the induced width plus one. Finding an optimal triangulation, one that minimizes the size of the largest clique, is NP-hard in general, but many heuristics work well in practice. For MAP inference, the same algorithm applies with sums replaced by maxes, producing max-marginals that reveal the most probable joint assignment.

I now give a compact, runnable Python illustration of the core operations on a tiny chordal graph. The script moralizes a simple directed model, forms maximal cliques by hand, builds a junction tree, assigns factors, and runs HUGIN-style separator updates to calibrate the tree. The numbers are small enough that a brute-force joint table can be computed independently and compared against the clique marginals.

```python
import itertools
import numpy as np

# Tiny Bayesian network: A -> B, A -> C, B -> D, C -> D
# Variables are binary. Factors: P(A), P(B|A), P(C|A), P(D|B,C)
A = np.array([0.6, 0.4])  # shape (a,)
B_A = np.array([[0.7, 0.3],   # A=0
                [0.2, 0.8]])  # A=1   shape (a,b)
C_A = np.array([[0.9, 0.1],
                [0.4, 0.6]])  # shape (a,c)
D_BC = np.array([[[0.8, 0.2],   # B=0,C=0
                  [0.6, 0.4]],  # B=0,C=1
                 [[0.4, 0.6],   # B=1,C=0
                  [0.1, 0.9]]]) # B=1,C=1  shape (b,c,d)

# Brute-force joint distribution over A,B,C,D
joint = A[:, None, None, None] * B_A[:, :, None, None] * C_A[:, None, :, None] * D_BC[None, :, :, :]
print("Joint shape:", joint.shape, "sum =", joint.sum())

# Moral graph cliques for this model: ABC and BCD
# Junction tree: ABC -- BC -- BCD
# Assign factors: P(A)*P(B|A)*P(C|A) to ABC; P(D|B,C) to BCD

# Build initial clique potentials by summing over variables not in the clique
# (in a real implementation factor assignment avoids this).
phi_ABC = np.einsum('a,ab,ac->abc', A, B_A, C_A)          # shape (a,b,c)
phi_BCD = D_BC.copy()                                      # shape (b,c,d)

# HUGIN-style calibration
# Message ABC -> BCD over separator BC
sep_BC = phi_ABC.sum(axis=0)           # marginal of ABC over BC
phi_BCD = phi_BCD * sep_BC[:, :, None]
mu_BC = sep_BC

# Message BCD -> ABC over separator BC
sep_BC_back = phi_BCD.sum(axis=-1)     # marginal of BCD over BC
phi_ABC = phi_ABC * (sep_BC_back / mu_BC)

# Normalize clique potentials
phi_ABC /= phi_ABC.sum()
phi_BCD /= phi_BCD.sum()

# Compare clique marginals to brute-force marginals
marg_ABC = joint.sum(axis=-1)
marg_BCD = joint.sum(axis=0)
print("ABC clique matches brute force:", np.allclose(phi_ABC, marg_ABC / marg_ABC.sum()))
print("BCD clique matches brute force:", np.allclose(phi_BCD, marg_BCD / marg_BCD.sum()))

# Variable marginals from calibrated cliques
print("P(A):", phi_ABC.sum(axis=(1, 2)))
print("P(D):", phi_BCD.sum(axis=(0, 1)))
print("Brute P(A):", joint.sum(axis=(1, 2, 3)))
print("Brute P(D):", joint.sum(axis=(0, 1, 2)))
```

This code demonstrates that the calibrated clique potentials agree with the exact joint marginals. In practice, the junction tree algorithm scales to much larger models as long as an appropriate triangulation keeps the maximal cliques small. It remains the foundational exact inference algorithm for probabilistic graphical models, underlying later variational and sampling methods that relax exactness for computational convenience.