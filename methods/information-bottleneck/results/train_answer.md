The method I am describing is the Information Bottleneck method. It addresses a basic representation-learning question: when an input variable X is rich and a target variable Y tells us what matters, how do we build a compressed summary T that keeps the information about Y while throwing away as much irrelevant detail about X as possible. The key move is to turn this informal goal into a variational tradeoff written entirely in information-theoretic quantities, so that the notion of "relevance" comes from the joint distribution p(x, y) rather than from a hand-crafted metric on X.

I start from the Markov chain Y -> X -> T. This ordering is important: T is produced from X, not from Y directly, and it must not introduce extra information about Y beyond what X already carries. The compression cost is measured by the mutual information I(X; T), which is the average number of bits (or nats, depending on the log base) that the code T carries about the input X. The value of the code is measured by I(T; Y), the information it preserves about the relevant variable Y. The Information Bottleneck objective is the Lagrangian

min_{p(t|x)}  I(X; T) - beta I(T; Y)

with beta >= 0. A small beta favors heavy compression and may collapse many inputs together; a large beta pushes the code to preserve more information about Y at the cost of a larger I(X; T). By sweeping beta, I trace an information-theoretic frontier that generalizes the rate-distortion curve, except that the distortion is not supplied externally. Instead, the distortion emerges from how well the predictive distribution p(y|t) approximates the input-level predictive distribution p(y|x).

To derive the stationary equations, I differentiate the Lagrangian with respect to a single encoder entry p(t|x), keeping the row constraint sum_t p(t|x) = 1 enforced by a multiplier. The derivative contains a log-assignment term plus a term proportional to the Kullback-Leibler divergence D_KL[p(y|x) || p(y|t)]. After absorbing the parts that depend on x but not on t into the row multiplier, the optimal assignment takes the rate-distortion-like form

p(t|x) = p(t) exp(-beta D_KL[p(y|x) || p(y|t)]) / Z(x, beta),

where Z(x, beta) normalizes over t. The codeword distributions themselves must be consistent with this encoder through Bayes' rule:

p(t) = sum_x p(x) p(t|x),
p(y|t) = sum_x p(y|x) p(t|x) p(x) / p(t).

These three equations together define a self-consistent fixed point. The distortion induced by the Information Bottleneck is therefore d_IB(x, t) = D_KL[p(y|x) || p(y|t)], which measures how much the predictive distribution associated with codeword t deviates from the predictive distribution associated with input x. Two inputs that make similar predictions about Y are encouraged to share the same codeword, even if their raw features are very different.

An equivalent way to organize the computation is to minimize the free energy

F = I(X; T) + beta E_{p(x,t)} D_KL[p(y|x) || p(y|t)].

Under the Markov chain Y -> X -> T, the expected KL distortion equals I(X; Y) - I(T; Y), so F differs from the Lagrangian only by the constant beta I(X; Y). Minimizing F is therefore the same problem, and it leads to the same alternating-minimization loop. Each substep, updating p(t), p(y|t), or p(t|x) while holding the others fixed, is convex in the distribution being updated, so the overall procedure monotonically decreases the free energy. The full problem is not jointly convex, which means initialization and the chosen cardinality of T can influence which fixed point is reached.

The boundary cases of the objective give useful intuition. At beta = 0, the exponential factor becomes one, so p(t|x) equals p(t) for every x. The code then carries no information about X, and with a minimal active alphabet it collapses to a single codeword. As beta grows, the KL term dominates, and each input is assigned to codewords whose predictive distributions over Y closely match p(y|x). The hard constraint on zero probabilities is also visible in the KL: if p(y|x) > 0 but p(y|t) = 0 for some y, the distortion is infinite, so that assignment receives zero probability.

In the finite-alphabet case, the algorithm is a direct fixed-point iteration. I choose the cardinality of T, initialize a row-stochastic encoder p(t|x), and alternate the three update equations until the encoder stabilizes. In practice it is safer to work in log space: compute log p(t) minus beta times the KL distortion, subtract a maximum for numerical stability, exponentiate, and normalize. After convergence I record I(X; T), I(T; Y), the expected distortion, and the free energy. Repeating the run for many values of beta traces the Information Bottleneck curve, which shows how much relevance can be preserved for each allowed level of compression.

The Information Bottleneck method has close ties to earlier work on distributional word clustering. In that setting, soft cluster memberships, averaged context distributions, KL mismatch, and annealing already produced useful hierarchical word representations. The Information Bottleneck lifts those ingredients into a general principle: the objects being averaged are the conditional distributions p(y|x) induced by the joint law, the cluster centroids are the conditional distributions p(y|t), and the annealing parameter is beta. This clarifies why the same mathematical structure appeared in clustering: it is the natural solution to a compression-versus-relevance tradeoff.

The method also makes a conceptual distinction that matters for representation learning. A predictor that minimizes prediction error can succeed while its internal summary still encodes many idiosyncratic details of X. A dimensionality-reduction method that only asks for small codes can keep the wrong details. The Information Bottleneck asks instead about information: how much about X survives, and how much about Y survives. The answer is a stochastic representation that is, in the limit of high beta, an approximate minimal sufficient statistic for Y given X. It is not trying to reconstruct X; it is trying to carry the part of X that is relevant for Y.

Because the objective is purely information-theoretic, it applies before any commitment to architecture, geometry, or feature distance. The only input required is the joint distribution p(x, y). This generality is also the source of the main practical limitation: exact computation requires knowledge of p(x, y), and for continuous or high-dimensional variables one must introduce approximations such as variational bounds or neural encoders. In the finite-alphabet case, though, none of that approximation machinery is needed, and the method reduces to a complete, self-contained protocol that I can state in full as the deliverable.

The protocol takes a finite joint table $p(x,y)$, a chosen code cardinality $|T|$, and a tradeoff parameter $\beta \ge 0$, and returns a stationary encoder by iterating the following fixed-point map. Start from any row-stochastic $p(t\mid x)$, and repeat the two half-steps

$$p(t) = \sum_x p(x)\, p(t\mid x), \qquad\qquad p(y\mid t) = \frac{1}{p(t)}\sum_x p(y\mid x)\, p(t\mid x)\, p(x),$$

$$p(t\mid x) = \frac{p(t)}{Z(x,\beta)}\exp\!\Big(-\beta\, D_{KL}\big[p(y\mid x)\,\big\|\,p(y\mid t)\big]\Big), \qquad Z(x,\beta) = \sum_t p(t)\exp\!\Big(-\beta\, D_{KL}\big[p(y\mid x)\,\big\|\,p(y\mid t)\big]\Big),$$

until the maximum entry-wise change in $p(t\mid x)$ falls below a chosen tolerance. Because each half-step is convex in the block it updates, the free energy $F = I(X;T) + \beta\, \mathbb{E}_{p(x,t)}D_{KL}[p(y\mid x)\|p(y\mid t)]$ decreases monotonically along this iteration and the map is guaranteed to reach a stationary point, though not necessarily the same one from every initialization. Reading $I(X;T)$ and $I(T;Y)$ off the converged $p(t\mid x)$, $p(t)$, and $p(y\mid t)$ gives one point on the Information Bottleneck curve; sweeping $\beta$ upward from $0$ traces the whole curve, from the trivial single-codeword solution at $\beta = 0$ to, as $\beta \to \infty$, codes that approximate a minimal sufficient statistic for $Y$ given $X$. That curve — the full compression/relevance frontier obtained by running this fixed point at every $\beta$, not any single value of it — is what the Information Bottleneck method delivers.
