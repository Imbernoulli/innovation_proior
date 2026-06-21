I am presenting rate–distortion theory, the part of information theory that extends the lossless source-coding theorem to approximate reproduction. Lossless coding gives entropy H(X) as the exact-reproduction limit, but that answer collapses as soon as exactness is relaxed: continuous sources would need infinitely many bits, and even discrete sources such as images or audio are usually acceptable only up to a tolerance. The right question is therefore not how many bits reproduce X exactly, but how many bits reproduce X to within a specified average distortion D.

To make the question precise, fidelity is modeled by a single-letter distortion d(x, x_hat) >= 0, and the per-block distortion is the average over symbols. A reproduction scheme is free to choose any joint distribution of source and reconstruction, equivalently a test channel q(x_hat | x), provided the expected distortion E[d(X, X_hat)] does not exceed D. The bits that the reconstruction must actually carry about the source are measured by mutual information I(X; X_hat), not merely by the entropy of X_hat, because I counts only the dependence on X and ignores irrelevant randomness. The rate–distortion function is therefore defined as the minimum mutual information over all test channels meeting the distortion budget:

R(D) = min_{q(x_hat|x) : E[d(X, X_hat)] <= D} I(X; X_hat).

This is the canonical object of rate–distortion theory.

The definition is the lossy mirror of channel capacity C = max_{p(x)} I(X; Y). Capacity maximizes mutual information over input distributions for a fixed channel, asking which source best matches a given noisy channel. Rate–distortion minimizes mutual information over channels for a fixed source, asking which reconstruction channel best matches the source under a fidelity budget. The two curves are dual extremal problems built from the same quantity.

The function R(D) has a clean shape. As the distortion budget grows, the feasible set of channels expands, so R(D) is non-increasing. It is convex in D because distortion is linear in the channel while mutual information is convex in the channel. At D = 0, when zero distortion forces exact reproduction for a discrete source, R(0) equals H(X). At the other extreme, if D is at least d_max = min_{x_hat} E[d(X, x_hat)], the best strategy is to output a constant reconstruction and R(D) drops to zero.

The operational meaning is given by the rate–distortion theorem. The converse shows no code can beat R(D). For any code of block length n with 2^{nR} codewords and average distortion D, the chain nR >= H(X_hat^n) >= I(X^n; X_hat^n) = sum_i I(X_i; X_hat_i) >= sum_i R(D_i) >= n R(D) forces R >= R(D). The achievability shows R(D) can be approached. Draw 2^{nR} reconstruction words independently from the output marginal induced by the minimizing test channel; for any R > R(D), a typical source block is distortion-covered with probability tending to one, so some codebook attains distortion close to D. Channel coding is sphere packing; rate–distortion coding is sphere covering.

The canonical closed-form examples confirm the theory. For a Gaussian source X ~ N(0, sigma^2) with squared error, R(D) = 1/2 log_2(sigma^2 / D) for 0 < D < sigma^2, R(0) = +infinity, and R(D) = 0 for D >= sigma^2. Equivalently D(R) = sigma^2 2^{-2R}, so each additional bit reduces squared error by a factor of four, about 6.02 dB per bit. For a binary source X ~ Bernoulli(p), p <= 1/2, with Hamming distortion, R(D) = H(p) - H(D) for 0 <= D <= p and zero thereafter; for p = 1/2 this curve is exactly the capacity of a binary symmetric channel with crossover D.

For parallel independent Gaussian components, the optimal allocation of a total distortion budget is reverse water-filling: every active component receives the same distortion level theta, and components whose variance lies below theta are discarded. This follows by minimizing the sum of component rate–distortion functions subject to the total distortion constraint.

Overall, rate–distortion theory replaces the single entropy number with a full curve that trades rate against distortion, supplies matching converse and achievability proofs, and yields concrete computations for the most important source–distortion pairs.

```python
import math


def H2(p):
    """Binary entropy in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def binary_rate_distortion(D, p=0.3):
    """Bernoulli(p) source with Hamming distortion."""
    D_max = min(p, 1 - p)
    if D >= D_max:
        return 0.0
    return H2(p) - H2(D)


def gaussian_rate_distortion(D, sigma2=1.0):
    """Gaussian source with squared-error distortion."""
    if D >= sigma2:
        return 0.0
    if D <= 0.0:
        return math.inf
    return 0.5 * math.log2(sigma2 / D)


if __name__ == "__main__":
    p = 0.3
    print(f"Binary source Bernoulli({p}), entropy H(p) = {H2(p):.4f} bits")
    for D in (0.05, 0.10, 0.15):
        R = binary_rate_distortion(D, p)
        print(f"  D = {D:.2f}: R(D) = {R:.4f}")

    print()
    print("Gaussian source N(0,1) with squared error")
    for D in (0.05, 0.10, 0.25):
        R = gaussian_rate_distortion(D)
        print(f"  D = {D:.2f}: R(D) = {R:.4f}")

    # One-bit scalar quantizer distortion for N(0,1) is (pi-2)/pi.
    scalar_one_bit = (math.pi - 2) / math.pi
    theoretical_one_bit = gaussian_rate_distortion(0.25)
    print()
    print(f"1-bit scalar quantizer distortion: {scalar_one_bit:.4f}")
    print(f"Theoretical distortion at R=1 bit: {0.25:.4f}")
    print(f"Theoretical R at scalar distortion: "
          f"{gaussian_rate_distortion(scalar_one_bit):.4f}")
```
