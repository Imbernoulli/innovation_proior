## Expert Systems Need Probabilities, Not Scores

Rule-based expert systems can describe domains with many uncertain findings and hidden causes, but
their early uncertainty mechanisms often use local scores rather than coherent probability. A fully
probabilistic model is attractive because every diagnosis, prediction, and observation has a common
semantic scale, yet exact calculation appears to demand a joint table over all variables.

## Factorization Is Local But Queries Are Global

A directed causal model factors the joint distribution into small conditional probability tables.
That representation is compact, but a posterior marginal still asks for a sum over all unobserved
variables. The graph says which pieces of the product are local; it does not immediately say how to
perform the global sum locally.

## Trees Expose The Separator Principle

On a tree, an edge separates the model into two sides. Once the variables on the edge interface are
fixed, everything on one side can be summarized and sent as a function of that interface. Two passes
are enough because information cannot leave a region and return through a second route.

## Cycles Break The Naive Local Program

In a graph with loops, the same variable can influence a calculation along multiple paths. Passing
tree-style summaries on the original graph risks either double-counting or prematurely summing out a
variable that is still needed elsewhere. Exactness requires separators that genuinely isolate the two
sides of each local calculation.

## Graph Theory Has A Candidate Language

Graph theory had already identified special finite graphs whose maximal complete sets behave well
under decomposition: chordless long cycles are absent, and clique intersections can act as coherent
interfaces. The open computational question is whether this graph-theoretic language can be made to
serve exact probabilistic calculation without destroying the compact local tables that made the model
usable in the first place.
