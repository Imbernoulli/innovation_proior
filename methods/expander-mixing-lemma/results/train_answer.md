The problem is to understand when a sparse regular graph behaves like a random graph in terms of edge distribution. Given a d-regular graph on n vertices, we would like to know whether the number of edges between any two subsets S and T is close to what we would expect in a random d-regular graph, which is roughly d|S||T|/n. A naive approach might try to verify this uniformity by checking every possible pair of subsets directly, but that is combinatorially impossible because there are exponentially many subsets. Another common temptation is to treat high connectivity or expansion as a substitute for randomness, but without a precise quantitative statement the intuition remains vague and hard to reuse across applications.

What is missing is a clean translation from spectral properties to combinatorial mixing. The key realization is that the deviation of edge counts from the random expectation can be captured by the quadratic form of the adjacency matrix on indicator vectors. Once the indicator vectors are decomposed into their uniform components and their orthogonal perturbations, the uniform part gives exactly the random expectation, and the orthogonal part captures all deviations. This decomposition allows a single global parameter, the second eigenvalue bound lambda, to control the mixing behavior for every pair of subsets simultaneously. Without this spectral reformulation, the notion of a graph "looking random" stays anecdotal and cannot be certified or transported to other problems.

The method is the Expander Mixing Lemma. It states that for a d-regular graph G with adjacency matrix A, if every eigenvalue of A other than the trivial eigenvalue d has absolute value at most lambda, then for every pair of vertex subsets S and T the number of ordered edges e(S,T) between S and T satisfies

|e(S,T) - d|S||T|/n| <= lambda sqrt(|S||T|).

The proof proceeds by writing the indicator vectors of S and T as the sum of their projections onto the all-ones vector and the remaining orthogonal components. The cross terms vanish because the all-ones vector is an eigenvector of A with eigenvalue d, leaving the random expectation as the main term and the quadratic form of the orthogonal components as the error. Since A acts on the orthogonal complement with operator norm at most lambda, the error is bounded by lambda times the product of the norms of the orthogonal components, which is at most lambda sqrt(|S||T|). This argument is short but powerful: it reduces an exponential collection of combinatorial constraints to one spectral condition.

The practical meaning is that a small second eigenvalue gives a reusable certificate of pseudorandomness. Instead of proving uniformity from scratch in each application, one checks the spectral gap once and then invokes the lemma to obtain uniform edge counts for free. This makes explicitly constructed sparse expanders usable as drop-in replacements for random graphs in contexts such as derandomization, sampling, network design, and complexity theory. The lemma is not a claim that expanders share every property of random graphs; it only certifies a specific but broadly useful form of edge-distribution uniformity, and it assumes regularity in its simplest form. Variants for irregular, weighted, or bipartite graphs require normalized Laplacians or adjusted definitions, but the core spectral-to-combinatorial translation remains the same.

```python
import numpy as np
from scipy.sparse.linalg import eigsh

def expander_mixing_lemma(adjacency_matrix, subsets_s, subsets_t):
    """
    Verify the Expander Mixing Lemma for a d-regular graph.

    Parameters
    ----------
    adjacency_matrix : np.ndarray, shape (n, n)
        Symmetric 0/1 adjacency matrix of a d-regular graph.
    subsets_s, subsets_t : list of iterables
        Lists of vertex subsets (each subset is an iterable of vertex indices).

    Returns
    -------
    max_ratio : float
        Maximum observed value of |e(S,T) - d|S||T|/n| / sqrt(|S||T|).
    nontrivial_spectral_bound : float
        Largest absolute value of a non-trivial eigenvalue.
    """
    A = np.asarray(adjacency_matrix, dtype=float)
    n = A.shape[0]
    degrees = A.sum(axis=1)
    d = float(degrees[0])
    if not np.allclose(degrees, d):
        raise ValueError("Graph must be d-regular.")

    # Compute eigenvalues and remove the trivial eigenvalue d.
    # The remaining eigenvalues are the non-trivial ones; lambda is their largest magnitude.
    k = max(1, min(n - 1, 12))
    eigenvalues = eigsh(A, k=k, which='LM', return_eigenvectors=False)
    nontrivial = eigenvalues[~np.isclose(eigenvalues, d, atol=1e-8)]
    lambda_bound = float(np.max(np.abs(nontrivial))) if len(nontrivial) > 0 else 0.0

    max_ratio = 0.0
    for S in subsets_s:
        for T in subsets_t:
            indicator_s = np.zeros(n)
            indicator_s[list(S)] = 1.0
            indicator_t = np.zeros(n)
            indicator_t[list(T)] = 1.0
            # Count ordered edges from S to T.
            e_st = float(indicator_s @ A @ indicator_t)
            expected = d * len(S) * len(T) / n
            deviation = abs(e_st - expected)
            denom = np.sqrt(len(S) * len(T))
            if denom > 0:
                ratio = deviation / denom
                if ratio > max_ratio:
                    max_ratio = ratio

    return max_ratio, float(lambda_bound)


# Example: a 3-regular Ramanujan-like graph on 8 vertices (cube graph).
if __name__ == "__main__":
    cube = np.zeros((8, 8), dtype=int)
    for i in range(8):
        cube[i, i ^ 1] = 1
        cube[i, i ^ 2] = 1
        cube[i, i ^ 4] = 1
    cube = np.maximum(cube, cube.T)  # symmetrize if needed

    subsets = [
        {0, 1, 2, 3},
        {0, 1},
        {4, 5, 6, 7},
        {0, 7},
    ]
    ratio, lam = expander_mixing_lemma(cube, subsets, subsets)
    print(f"Max observed |deviation|/sqrt(|S||T|): {ratio:.4f}")
    print(f"Nontrivial eigenvalue bound lambda:    {lam:.4f}")
    print(f"Lemma guarantees ratio <= lambda:      {ratio <= lam + 1e-9}")
```
