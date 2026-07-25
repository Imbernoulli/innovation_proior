## Problem background

The object of study is the relationship between sparse regular graphs and random graphs in terms of edge distribution. Given an n-vertex, d-regular graph G, people care about whether its edges are "spread evenly" among subsets of vertices: in a random d-regular graph, the number of edges between two subsets S,T is roughly d|S||T|/n. A natural question is: for a specific, deterministically constructed regular graph, how closely can its edge count approach this random expectation.

The edge count itself is counted using ordered endpoints: e(S,T) counts ordered pairs where one endpoint lies in S and the other in T; when S and T intersect, internal edges are counted twice.

## Linear algebra perspective

Writing the sets S,T as indicator vectors 1_S,1_T, the number of edges between the subsets is exactly the quadratic form 1_S^T A 1_T of the adjacency matrix A. This hands the combinatorial quantity directly over to linear algebra.

For a d-regular graph, the all-ones vector is an eigenvector of A with eigenvalue d, which is its largest eigenvalue, called the trivial eigenvalue. All other eigenvalues correspond to eigenvectors orthogonal to the all-ones vector, and when sorted by absolute value, the second largest one (usually denoted lambda or lambda_2) characterizes the graph's spectrum outside the trivial direction. The spectrum is used in many settings to measure a graph's connectivity and "expansion": the smaller the second eigenvalue, the closer the graph is to being a highly connected, rapidly mixing object.

## Pseudorandomness and construction

Random d-regular graphs behave well in many respects — uniform edge distribution, strong connectivity, fast mixing — but they are random, and hard to write down explicitly. Many applications (bounds for independent sets and colorings, sampling, network robustness, complexity theory, pseudorandom generation) want deterministic, explicitly constructible graphs that still inherit some of the good properties of random graphs.

Explicitly constructing regular graphs with small second eigenvalue (expander graphs) is an active topic in this direction. The core question of interest here is: what quantitative connection can be established between a regular graph's spectral information (particularly d and the second eigenvalue lambda) and its combinatorial properties (specifically, the edge count between arbitrary pairs of subsets)?
