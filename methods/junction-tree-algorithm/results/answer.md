# Junction Tree Algorithm

Given a Bayesian network or factorized undirected model, exact inference is obtained by compiling the
model into a chordal clique tree and calibrating local potentials.

1. If the model is directed, moralize it: connect co-parents and drop arrow directions.
2. Triangulate the undirected graph by adding fill-in edges until every cycle of length at least four has a chord.
3. Form the maximal cliques of the triangulated graph.
4. Connect the cliques into a tree satisfying running intersection: for every variable X, the cliques containing X form a connected subtree. A maximum-weight spanning tree over clique intersections is one common construction.
5. Assign each original factor to a clique containing its scope and multiply assigned factors into clique potentials.
6. Absorb hard evidence by restricting affected potentials to the observed states, leaving the normalizing constant implicit; soft or virtual evidence is multiplied in as an additional local potential.
7. Calibrate by passing messages over separators S_ij = C_i cap C_j.

For the HUGIN/pgmpy cached separator update, let beta_i be the current belief at sending clique C_i and let mu_ij be the stored separator belief:

```text
sigma_{i->j}(S_ij) = sum_{C_i \ S_ij} beta_i(C_i)
beta_j <- beta_j * sigma_{i->j} / mu_ij    if mu_ij is already stored
beta_j <- beta_j * sigma_{i->j}            otherwise
mu_ij <- sigma_{i->j}
```

After any valid collect/distribute message schedule has calibrated the tree, adjacent cliques agree on every separator:

```text
sum_{C_i \ S_ij} beta_i = sum_{C_j \ S_ij} beta_j = mu_ij
```

The original Lauritzen-Spiegelhalter architecture reaches the same clique-marginal target by two passes through a set-chain representation. In the later HUGIN architecture, the division is cached in separator storage; the pgmpy artifact follows this cached separator form. For max-calibration/MAP, replace each separator sum above by a max and check equality of max-marginals.

The exactness comes from chordality plus running intersection. Chordality gives clique scopes large enough to contain all factors created by exact elimination; running intersection ensures separator consistency glues those local clique beliefs into one joint distribution. Once calibrated, any variable marginal is obtained by marginalizing any calibrated clique that contains the variable.

The cost is controlled by the largest clique created by triangulation. For tabular variables, storage and arithmetic are exponential in that clique size, i.e. exponential in induced width/treewidth plus one.
