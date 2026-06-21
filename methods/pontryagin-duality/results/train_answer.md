I propose that the canonical name for this result be Pontryagin duality for locally compact abelian groups. The theorem unifies the many Fourier transforms that appear across analysis: Fourier series on the circle, the Fourier transform on the real line, and character tables of finite abelian groups are all instances of the same structural principle rather than unrelated formulas.

The starting observation is that every familiar Fourier object has the same shape. On the circle group T, the basic oscillations are z ↦ z^n indexed by integers n ∈ Z. On the integers Z, the oscillations are n ↦ exp(2π i n θ) indexed by θ ∈ T. On the real line R, the oscillations are x ↦ exp(2π i ξ x) indexed by ξ ∈ R. In a finite abelian group, the oscillations are the rows of a character table. In each case the probe is a continuous homomorphism from the group into the circle group T = {z ∈ C : |z| = 1}. Such a probe is called a continuous character. The collection of all continuous characters is the Pontryagin dual of G, written G^ = Hom_cont(G, T). Pointwise multiplication makes G^ into an abelian group, and the compact-open topology makes it into a locally compact abelian group in its own right.

The compact-open topology is not an arbitrary choice. A character is a continuous function on G, and local compactness tells us that compact subsets are the right controlled pieces of G. A neighborhood of the trivial character in G^ asks that a character map a given compact set K ⊆ G into a small neighborhood of 1 in T. When G is discrete, compact sets are finite, so the dual sits as a closed subgroup of the product T^G and is therefore compact. When G is compact, the whole group is a test set, and any nontrivial character is bounded away from 1, so every character is isolated and the dual is discrete. This explains why Fourier series have discrete frequencies on compact domains and why transforms on discrete domains have compact frequency spaces.

The theorem is completed by the evaluation map e_G : G → G^^, defined by e_G(x)(χ) = χ(x). Each point x ∈ G gives a character on the dual group because evaluation at a fixed point is continuous and respects multiplication of characters. Pontryagin duality asserts that this natural map is an isomorphism of topological groups: every locally compact abelian group is canonically recovered from its continuous circle-valued characters.

Injectivity relies on the fact that characters separate points. If x ≠ 0 in G, there exists a continuous character χ with χ(x) ≠ 1. For finite abelian groups this follows from splitting off a cyclic quotient; for R it follows from the exponential functions; and the general locally compact abelian case reduces to these building blocks through compact subgroups, discrete quotients, and Euclidean pieces. Surjectivity is the deeper part: every continuous character Φ : G^ → T is evaluation at some x ∈ G. The continuity condition prevents Φ from depending on an uncontrolled infinite pattern of probes, forcing it to be a single-point evaluation. Once e_G is bijective, the compact-open topology ensures it is also a homeomorphism, so the original topology of G is exactly the topology induced from G^^.

The Fourier transform then appears in its proper structural role. For a Haar-integrable function f on G, the transform is f^(χ) = ∫_G f(x) χ(x)̄ dx, with χ ranging over G^. The frequency variable is not merely a parameter inserted into an integral; it is a point of the dual group. Translation on G becomes multiplication by characters on G^, convolution becomes pointwise multiplication, and compactness trades with discreteness. The classical formulas are special cases: T^ ≅ Z gives Fourier series, Z^ ≅ T gives the discrete-time Fourier transform, and R^ ≅ R gives the ordinary Fourier transform, where the self-duality hides the group-theoretic symmetry because the two sides look identical.

The insight I want to emphasize is that the integral formula is the analytic shadow of a deeper duality. The source of Fourier analysis is the statement that a locally compact abelian group can be reconstructed from its continuous unitary characters. This is Pontryagin duality.

The following Python script illustrates the theorem for finite cyclic groups. For Z_n, the characters are χ_k(j) = exp(2π i k j / n), indexed by k ∈ Z_n. The dual group is isomorphic to Z_n, and the double-dual evaluation map sends an element j ∈ Z_n to the function k ↦ χ_k(j), which is exactly the character χ_j of the dual. The script builds the character table, verifies orthogonality, and checks that the double-dual identification holds.

```python
import numpy as np


def character_table(n):
    """Return the n x n character table of Z_n.
    Entry [k, j] is chi_k(j) = exp(2*pi*i*k*j/n)."""
    j = np.arange(n)
    k = np.arange(n).reshape(-1, 1)
    return np.exp(2j * np.pi * k * j / n)


def main():
    n = 8
    chi = character_table(n)

    # Orthogonality of distinct characters: rows have zero inner product.
    for k1 in range(n):
        for k2 in range(n):
            inner = np.vdot(chi[k1], chi[k2]) / n
            expected = 1.0 if k1 == k2 else 0.0
            assert np.isclose(inner, expected), f"orthogonality failed at {k1},{k2}"

    # Pontryagin double dual: evaluation at j reproduces character chi_j on the dual.
    for j in range(n):
        double_dual_row = chi[:, j]          # k -> chi_k(j)
        expected_row = chi[j % n, :]         # chi_j on the dual group
        assert np.allclose(double_dual_row, expected_row), f"double dual failed at {j}"

    # Fourier inversion on a sample function f: Z_n -> C.
    rng = np.random.default_rng(0)
    f = rng.normal(size=n) + 1j * rng.normal(size=n)
    f_hat = chi @ f / n                     # Fourier coefficients
    f_recovered = np.conj(chi).T @ f_hat    # inverse transform
    assert np.allclose(f, f_recovered)

    print(f"Pontryagin duality verified for Z_{n}.")
    print("Character table (phases in multiples of pi):")
    print(np.round(np.angle(chi) / np.pi, 2))


if __name__ == "__main__":
    main()
```
