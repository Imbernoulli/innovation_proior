I begin by observing that the central problem of supervised learning is not merely to fit the training sample, but to choose a predictor whose expected loss on future draws from the unknown distribution remains small. The expected risk is the integral of the loss against the true data-generating measure, while the only quantity I can evaluate directly is the empirical average over the observed sample. A naive strategy would minimize the empirical risk alone, yet that ignores the fact that the hypothesis I select is itself a function of the sample. Once the data participates in the choice, a concentration statement that holds for one fixed function no longer applies automatically; I need uniform control over the entire class of candidate predictors.

This observation leads me to organize the candidate functions into a nested structure of hypothesis spaces, each equipped with a capacity measure such as a VC dimension or a comparable uniform-convergence guarantee. I denote the structure by S_1 subset S_2 subset ... subset S_n, with corresponding capacities h_1 < h_2 < ... < h_n. Inside each space I solve ordinary empirical risk minimization, producing a candidate alpha_k. The decisive step is not to keep the candidate with the smallest training error, because that candidate typically comes from the richest and most over-capacity class. Instead I choose the level that minimizes a finite-sample upper bound on the true risk, namely the empirical risk of the level's best candidate plus a confidence term that grows with the capacity of the level and shrinks with the sample size.

For binary classification, Vapnik's classical VC bound supplies the additive confidence term Omega_0(h, l, eta) = sqrt((h(ln(2l/h)+1) - ln eta) / l), so that with probability at least 1 - eta the true error of any predictor in a class of VC dimension h is bounded by its training error plus Omega_0. Applying this logic level by level gives the structural risk minimization principle: select k* = argmin_k [R_emp(alpha_k) + Omega(h_k, l, eta_k)], and return alpha_{k*}. The selected predictor therefore trades the approximation error of a small class against the estimation error of a large class.

When the structure contains countably many classes, I must ensure that all bounds hold simultaneously. I assign prior weights w(k) with sum_k w(k) <= 1 and set eta_k = w(k) eta for each level. The union bound then guarantees that with probability at least 1 - eta, every level satisfies its own bound, so the selected level inherits an oracle-style guarantee against the best predictor in each candidate class plus that class's confidence penalty.

The principle is not merely a penalty on parameter count. Capacity measures such as VC dimension reflect the class's ability to realize dichotomies, not the number of scalar parameters. A concrete illustration is the margin bound for hyperplane classifiers: if inputs lie in a ball of radius R and a hyperplane separates them with margin Delta, the VC dimension is bounded by min(R^2 / Delta^2, n) + 1. Maximizing the margin therefore reduces capacity. A hard-margin support vector machine in the separable case keeps empirical error zero while selecting the separator from the smallest-capacity margin class compatible with the data; soft-margin and kernel variants preserve the same tradeoff between empirical violations and capacity control.

The canonical name of the method is Structural Risk Minimization, and its essential message is that model selection should be driven by a finite-sample guarantee rather than by training loss alone. Stated in full, the procedure is this. Fix a nested structure of hypothesis spaces

$$S_1 \subset S_2 \subset \cdots \subset S_n, \qquad h_1 < h_2 < \cdots < h_n,$$

where each $h_k$ is the capacity of $S_k$ — a VC dimension or a comparable uniform-convergence measure. Inside every level, solve ordinary empirical risk minimization,

$$\alpha_k = \arg\min_{\alpha \in S_k} R_{\mathrm{emp}}(\alpha).$$

Attach to each level the confidence term that the VC bound supplies; for binary classification this is

$$\Omega_0(h, l, \eta) = \sqrt{\frac{h\big(\ln(2l/h) + 1\big) - \ln \eta}{l}},$$

so that with probability at least $1-\eta$ the true error of any predictor drawn from a class of VC dimension $h$ satisfies $P(w) < v(w) + \Omega_0(h, l, \eta)$. When the structure has countably many levels, assign prior weights $w(k)$ with $\sum_k w(k) \le 1$ and set $\eta_k = w(k)\eta$ for each level, so that a single union bound makes every level's guarantee hold simultaneously with overall probability at least $1-\eta$. Finally, select the level whose upper bound on true risk is smallest,

$$k^* = \arg\min_k \Big[ R_{\mathrm{emp}}(\alpha_k) + \Omega(h_k, l, \eta_k) \Big], \qquad \text{return } \alpha_{k^*}.$$

The returned predictor $\alpha_{k^*}$ then carries the oracle-style guarantee that, at every level $k$ simultaneously, it is within $\Omega(h_k, l, \eta_k)$ of that level's own best candidate — the finite-sample statement that a rule minimizing training loss alone can never make, because that rule has no mechanism for charging a class for its capacity to fit the sample by chance.
