## Problem Setting

The input is an undirected graph together with two named vertices, and the question is whether a path connects them. Ordinary graph search answers quickly but stores a visited set or frontier, so its memory use is far larger than the name of a single vertex.

## Memory Barrier

A deterministic recursive reachability method can trade time for space by asking whether a midpoint splits a bounded-length path. This gives a quadratic logarithmic-space upper bound and applies even to directed reachability.

## Random-Walk Baseline

In undirected graphs, a random walk from one vertex reaches every vertex in its connected component within polynomially many steps with high probability. That gives a logarithmic-space randomized algorithm: store the current vertex and a step counter, while randomness supplies the next edge choices.

## Expansion Background

A sparse regular graph with strong spectral expansion has logarithmic diameter. In such a graph, the number of distinct logarithmically long edge-label sequences is polynomial in the number of vertices.

## Local Graph Interfaces

Regular edge-labelled graphs can be accessed by a local port operation that maps a vertex and outgoing edge label to the neighboring vertex and the return label. This representation lets graph powers and graph products be described as local manipulations of vertex names and port labels, without storing a full adjacency table.
