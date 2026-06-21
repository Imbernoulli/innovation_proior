## Expert Systems Need Probabilities, Not Scores

Rule-based expert systems can describe domains with many uncertain findings and hidden causes, but
their early uncertainty mechanisms often use local scores rather than coherent probability. A fully
probabilistic model is attractive because every diagnosis, prediction, and observation has a common
semantic scale, yet exact calculation appears to demand a joint table over all variables.

## Factorization Is Local But Queries Are Global

A directed causal model factors the joint distribution into small conditional probability tables.
That representation is compact, but a posterior marginal still asks for a sum over all unobserved
variables.

## Trees Expose The Separator Principle

On a tree, an edge separates the model into two sides. Once the variables on the edge interface are
fixed, everything on one side can be summarized and sent as a function of that interface. Two passes
are enough because information cannot leave a region and return through a second route.

## Cycles Complicate Local Computation

In a graph with loops, the same variable can influence a calculation along multiple paths. Passing
tree-style summaries on the original graph risks double-counting when the same evidence reaches a
node by two routes simultaneously.

## Graph Theory Has A Candidate Language

Graph theory had already identified special finite graphs whose maximal complete sets behave well
under decomposition: chordless long cycles are absent, and clique intersections form coherent
interfaces between parts of the graph. The computational question is how to organize exact
probabilistic inference on structured models of this kind.
