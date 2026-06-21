I will explain Morse theory as a way to reconstruct the global topology of a smooth manifold by watching how the sublevel sets of a generic real-valued function evolve. Let M be a smooth n-dimensional manifold and let f : M -> R be a smooth function. For each real number a I write M^a = { x in M : f(x) <= a }. The idea is to start with a very negative and let it increase. As the sublevel set grows, its topology can change only at special values of a. Morse theory tells us exactly what those special values are and how the topology changes at each one.

The first observation is that most values are not special at all. Suppose the band f^{-1}([a,b]) is compact and contains no critical points, meaning df is nonzero everywhere in that band. Choose a Riemannian metric on M. Because df is nonzero, the gradient grad f is also nonzero. I want a vector field whose flow moves points uphill at unit speed with respect to f, so I normalize by the squared norm and set X = grad f / ||grad f||^2. Then df(X) = <grad f, grad f> / ||grad f||^2 = 1. Compactness guarantees the flow exists for the finite time needed. If phi_t is the flow of X, then f(phi_t(x)) = f(x) + t as long as the trajectory stays in the band. Flowing for time b - a identifies level f^{-1}(a) with f^{-1}(b), and the entire band is product-like. With a smooth cutoff that vanishes below a and agrees with X inside the band, this gives a diffeomorphism from M^a to M^b. Reversing the flow gives a deformation retraction of M^b onto M^a. So between critical values nothing topological happens; the sublevel sets are all the same up to diffeomorphism.

This means all the interesting information is concentrated at critical points, where df vanishes. At a critical point p the Hessian is a well-defined quadratic form on the tangent space T_p M, independent of coordinates up to the usual change of basis. If the Hessian is nonsingular, the critical point is called nondegenerate. The index lambda is the number of negative eigenvalues of the Hessian, counted with multiplicity. Nondegenerate critical points are the useful ones because they are isolated and stable under small perturbations. A degenerate critical point such as f(x) = x^3 in one dimension has zero Hessian and its local topology is not determined by a single integer; a small perturbation can remove it or split it into nondegenerate points. For a generic smooth function on a compact manifold, all critical points are nondegenerate, and their critical values can be separated.

The local model for a nondegenerate critical point is given by the Morse lemma. It says that after a smooth change of coordinates centered at p, the function looks exactly like f(u,v) = c - ||u||^2 + ||v||^2, where c = f(p), u lies in R^lambda, and v lies in R^{n-lambda}. The number lambda is precisely the Hessian index. This is much stronger than a Taylor approximation: the higher-order terms are completely removed by the coordinate change. The local picture is therefore an exact quadratic saddle, with lambda descending directions and n - lambda ascending directions.

Now consider a thin band around a critical value c where there is exactly one nondegenerate critical point p of index lambda. Outside a small coordinate neighborhood of p, the function has no critical points in this band, so the regular-band argument applies there and contributes only a product. Inside the coordinate neighborhood the sublevel condition changes from -||u||^2 + ||v||^2 <= -epsilon just below c to -||u||^2 + ||v||^2 <= +epsilon just above c. Below c the central disk in the negative directions is missing, because ||v||^2 <= ||u||^2 - epsilon forces u to be bounded away from zero. Above c that disk appears. More precisely, for suitable radii r and R with r^2 < epsilon and R^2 - r^2 > epsilon, the set H = { (u,v) : ||u|| <= R, ||v|| <= r } is a lambda-handle D^lambda x D^{n-lambda} sitting inside the upper local sublevel set. Its attaching region S^{lambda-1} x D^{n-lambda} lies in the lower local sublevel set. The rest of the upper local sublevel set can be pushed back down by the gradient flow of the quadratic form. Therefore M^{c+epsilon} is obtained from M^{c-epsilon} by attaching a lambda-handle. Collapsing the D^{n-lambda} factor of the handle shows that, up to homotopy, crossing the critical value attaches a single lambda-cell.

This gives a complete reconstruction recipe. Start below the minimum value of f. As a increases, each time a passes a critical value the sublevel set changes by attaching a handle whose dimension equals the index of the corresponding critical point. Between critical values the sublevel set does not change. If M is compact and all critical values are distinct, then proceeding through the critical values in order yields a handle decomposition of M. After collapsing handles along their descending or ascending factors, one obtains a CW complex with one cell of dimension lambda for every critical point of index lambda. The same description covers minima, saddles, and maxima without changing the language: index 0 creates a new component, index 1 creates a tunnel or merges components, and index n caps off the manifold.

The method I have described is Morse theory. It converts the smooth structure of a manifold into a combinatorial skeleton by filtering through a generic smooth function. The two pillars are normalized gradient flow for regular bands and the Morse lemma for nondegenerate critical points. Together they show that the only topological events in the filtration are handle attachments, and the dimensions of those handles are read directly from the Hessian indices.

```python
import numpy as np

# Morse theory illustration on the 2-torus.
# We use f(x,y) = cos(2*pi*x) + cos(2*pi*y) on the unit square with periodic
# boundary conditions. Its critical points are found analytically, classified
# by the Hessian index, and the alternating sum of critical counts is compared
# to the Euler characteristic of the torus, which is 0.

def f(x, y):
    return np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)

def grad_f(x, y):
    return np.array([
        -2 * np.pi * np.sin(2 * np.pi * x),
        -2 * np.pi * np.sin(2 * np.pi * y)
    ])

def hess_f(x, y):
    c = (2 * np.pi) ** 2
    return np.array([
        [-c * np.cos(2 * np.pi * x), 0.0],
        [0.0, -c * np.cos(2 * np.pi * y)]
    ])

# Critical points of f in [0,1)^2 come from sin(2*pi*x)=0 and sin(2*pi*y)=0.
critical_points = [
    (0.0, 0.0),
    (0.0, 0.5),
    (0.5, 0.0),
    (0.5, 0.5)
]

# Classify each critical point by the number of negative Hessian eigenvalues.
index_counts = [0, 0, 0]
print("Critical point (x, y) -> Morse index")
for x, y in critical_points:
    H = hess_f(x, y)
    eigvals = np.linalg.eigvalsh(H)
    index = int(np.sum(eigvals < 0))
    index_counts[index] += 1
    print(f"  ({x:.1f}, {y:.1f}) -> eigenvalues {eigvals} -> index {index}")

# Euler characteristic from Morse counts: sum (-1)^index * count.
euler = sum((-1)**k * index_counts[k] for k in range(3))
print("Morse index counts (index 0 / 1 / 2):", index_counts)
print("Computed Euler characteristic:", euler)
print("Expected Euler characteristic of the 2-torus: 0")
assert euler == 0, "Morse counts should give Euler characteristic 0 for the torus."
print("Morse-theory check passed.")
```
