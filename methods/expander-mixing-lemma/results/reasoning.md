Let G be a d-regular graph on n vertices with adjacency matrix A. Because G is d-regular, the all-ones vector is an eigenvector of A with eigenvalue d. This direction represents completely uniform edge density: if we only look at the projection of the set indicator vectors onto the all-ones direction, there should be d|S||T|/n ordered edges between S and T.

For arbitrary sets S,T, decompose the indicator vectors into a uniform part and an orthogonal perturbation:

1_S = (|S|/n)1 + x_S, where x_S is orthogonal to 1;
1_T = (|T|/n)1 + x_T, where x_T is orthogonal to 1.

Substitute into the edge-count expression e(S,T)=1_S^T A 1_T. Since A1=d1, the cross terms vanish, and the main term is exactly d|S||T|/n, with the error term being x_S^T A x_T.

If all nontrivial eigenvalues of A have absolute value at most lambda, then A restricted to the orthogonal complement of 1 has operator norm at most lambda. Hence

|x_S^T A x_T| <= lambda ||x_S|| ||x_T|| <= lambda sqrt(|S||T|).

This is the proof skeleton of the Expander Mixing Lemma. Its power lies in the fact that all set-level deviations of edge distribution are controlled by this same single spectral parameter. Intuitively, a random graph would not let any subset direction retain a large structural deviation; in linear-algebra terms this corresponds exactly to the nontrivial eigen-directions having no large eigenvalue.

From the perspective of pseudorandomness, this lemma turns the construction problem into a spectral design problem. We don't need to directly enumerate exponentially many subsets to verify uniformity; it suffices to prove that the graph's second eigenvalue is small, and we get a mixing guarantee for every pair of subsets. This lets explicitly constructed sparse graphs substitute for random graphs in many applications, because applications usually only need some quantifiable uniform edge distribution, not a genuinely independent random generative process.

So the transferability of the Expander Mixing Lemma comes from a clean interface: spectral gap in, combinatorial uniformity out. Different fields can each make use of the same output property — for example, to rule out large independent sets, to prove the nonexistence of small cuts, to control the sampling bias of random walks, or to provide a substitute structure with bounded randomness in complexity theory. It turns "looking random" into "random-like behavior certified by a quadratic-form bound."
