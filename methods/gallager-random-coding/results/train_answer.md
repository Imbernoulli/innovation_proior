Shannon's noisy-channel coding theorem settles the qualitative question: below capacity C reliable communication is possible, above it impossible. But it says almost nothing quantitative. If I fix a rate R below C and ask how long a block length N I need to reach a target error probability, the theorem only promises that eventually the error goes to zero. Two channels can have the same capacity yet very different practical performance, and the theorem gives no handle for comparing them. Existing approaches sharpen this in limited ways. The union bound paired with the Bhattacharyya distance gives a clean exponential bound, but it is tied to a single exponent that reaches zero at the cutoff rate R0, which is strictly below C, so it says nothing in the high-rate regime that matters most. Fano's general bounds for arbitrary discrete memoryless channels do reach up to C, but the derivation is long and intricate, and the resulting exponent is hard to compute or interpret. Elias gave exact exponents for the binary symmetric channel, yet the construction does not transfer cleanly to other channels or to continuous ones like the additive Gaussian channel. What is missing is a single simple argument that produces the whole error-exponent curve E(R) for any discrete memoryless channel and extends naturally to continuous inputs with constraints.

The method that closes this gap is the Gallager random-coding exponent. Start with maximum-likelihood decoding for a random codebook: each of the M = e^{NR} codewords is drawn independently letter by letter from some input distribution p. The ML decoder fails for message m when some competitor codeword x_{m'} has likelihood P(y | x_{m'}) at least as large as P(y | x_m). The naive way to bound this failure event is a union bound over competitors, each bounded by the Bhattacharyya tilt with exponent one half. That is the special case rho = 1 of a one-parameter family. Instead, overbound the failure indicator by the ratio of the sum of competitor likelihoods raised to 1/(1+rho) over the sent-word likelihood raised to 1/(1+rho), then raise the whole ratio to the power rho. This bound is always nonnegative, is at least one whenever a competitor wins, and leaves rho as a free parameter. Substituting into the error probability and averaging over the ensemble exploits the independence of codewords: the factor depending on the sent word separates from the factor depending on the competitors. Because xi^rho is concave for 0 < rho <= 1, Jensen's inequality pulls the ensemble average inside the competitor sum. Everything then collapses, for a memoryless channel, into a single-letter expression.

Define E0(rho, p) = -ln sum_j ( sum_k p_k P_{jk}^{1/(1+rho)} )^{1+rho}. The averaged ML error for any message is bounded by exp[-N (E0(rho, p) - rho R)], valid for every rho in [0,1] and every input distribution p. Optimizing these free parameters gives the random-coding exponent E_r(R) = max_{0 <= rho <= 1, p} [E0(rho, p) - rho R], so that there exists a length-N rate-R code with P_e <= exp[-N E_r(R)]. The reason this improves on the union bound is that at rho = 0 the inner sum becomes the output marginal, so E0(0,p) = 0, and the derivative of E0 at rho = 0 equals the mutual information I(p). Thus for small rho the exponent behaves like rho (I(p) - R), which is positive whenever R < I(p). Choosing p at capacity gives I(p) = C, so E_r(R) > 0 for every R < C. The union bound failed because it was frozen at rho = 1; allowing rho to shrink toward zero produces gentler supporting lines whose slope at the origin is exactly the mutual information, carrying the exponent all the way to capacity. E0 is increasing and concave in rho, so the curve is well behaved: for high rates an interior rho solves R = partial E0 / partial rho, and for low rates the optimum sits at rho = 1 giving a straight line whose intercept is the cutoff rate.

At low rates the averaged bound is slightly loose because rare codebooks where two messages receive nearly identical codewords dominate the average. The expurgated bound fixes this by discarding the worst half of the codewords, costing only (ln 2)/N in rate. It yields a second exponent E_x(rho, p) - rho R for rho >= 1, where E_x uses pairwise Bhattacharyya distances, and it tightens the low-rate end. Fano's converse lower bound uses the same E0 functional but with rho ranging over all positive values; the achievability and converse coincide for rates between the critical rate and capacity, so in that band E_r(R) is the exact reliability function of the channel. The same argument extends to input constraints by tilting the ensemble with e^{r f(x)}; for the additive Gaussian channel the optimal input is Gaussian and the bound reproduces Shannon's exact Gaussian exponent.

```python
import numpy as np

def E0(rho, p, P):
    """Gallager single-letter function.
    P[j, k] = Pr(output b_j | input a_k); p[k] input probability."""
    inner = (p[None, :] * P ** (1.0 / (1.0 + rho))).sum(axis=1)
    return -np.log((inner ** (1.0 + rho)).sum())

def E_r(R, P, p_grid, rho_grid):
    """Random-coding exponent E_r(R) = max_{rho in [0,1], p} [E0(rho,p) - rho*R]."""
    return max(E0(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)

def E_x(rho, p, P):
    """Expurgated single-letter function, valid for rho >= 1."""
    pair = np.sqrt(P[:, :, None] * P[:, None, :]).sum(axis=0)
    return -rho * np.log((p[:, None] * p[None, :] * pair ** (1.0 / rho)).sum())

def E_ex(R, P, p_grid, rho_grid):
    """Expurgated exponent for low rates."""
    return max(E_x(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)

def error_probability_bound(N, R, P, p_grid, rho_grid):
    """There exists a length-N rate-R code with ML error <= exp(-N E_r(R))."""
    return np.exp(-N * E_r(R, P, p_grid, rho_grid))

def mutual_information(p, P):
    """I(p) = E0'(0), the slope that keeps E_r(R) positive for R < C."""
    q = P @ p
    return sum(p[k] * P[j, k] * np.log(P[j, k] / q[j])
               for k in range(len(p)) for j in range(P.shape[0])
               if p[k] > 0 and P[j, k] > 0)
```
