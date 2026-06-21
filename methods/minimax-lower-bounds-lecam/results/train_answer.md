I will describe the canonical method known as Le Cam's two-point method for minimax lower bounds, which is the simplest version of the Le Cam lower bound technique in statistical decision theory. The goal of a minimax lower bound is to show that, no matter which estimator we pick before seeing data, nature can choose a distribution inside the model class that forces the expected loss to be at least some positive quantity. Le Cam's idea is that we do not have to fight the entire model class at once. Instead, we embed a hard binary testing problem inside the estimation problem and show that any estimator accurate enough would also solve that test, which information theory tells us is impossible.

The construction begins by selecting two admissible distributions P0 and P1 inside the model class. These two worlds are chosen so that their target quantities are far apart in the metric that defines the loss. Concretely, if theta(P) is the parameter or functional of interest and rho is the loss metric, we require rho(theta(P0), theta(P1)) to be at least 2s for some positive accuracy radius s that we want to rule out. This separation means that no single point in the parameter space can be within distance s of both theta(P0) and theta(P1), because two points separated by 2s cannot share a common neighbor of radius s.

Given any estimator hat theta, I can define an induced binary test psi that looks at the estimate and chooses whichever target value is closer under rho. Formally, psi(X) is the index i in {0, 1} that minimizes rho(hat theta(X), theta(Pi)). Now suppose the estimator is uniformly accurate to radius s under both P0 and P1. Then under P0 the estimate must lie within distance s of theta(P0), so the induced test would correctly return 0. Under P1 the estimate must lie within distance s of theta(P1), so the induced test would correctly return 1. Therefore a uniformly accurate estimator gives a binary test with small total error.

Le Cam's method reverses this chain of reasoning. For two simple hypotheses, the smallest possible sum of type I and type II errors over all measurable tests is exactly 1 minus the total variation distance between the n-sample laws, that is, inf_psi [P0^n(psi=1) + P1^n(psi=0)] = 1 - TV(P0^n, P1^n). This identity comes from the Neyman-Pearson lemma, and it means no test can do better than the total-variation limit. Combining the two pieces, if the testing error is bounded below by 1 - TV(P0^n, P1^n), then the estimation risk must also be bounded below. For a squared-error type loss scaled by s^2, a standard form is inf over estimators hat theta of sup over P in the class of E_P[rho(hat theta, theta(P))^2] is at least (s^2 / 2) [1 - TV(P0^n, P1^n)].

In practice, total variation can be awkward to compute directly for product measures or for mixture constructions, so the method is usually applied through surrogates. The Kullback-Leibler divergence is useful because Pinsker-type inequalities translate a small KL into a small total variation, which in turn keeps 1 - TV away from zero. Hellinger distance is also popular because its affinity tensorizes cleanly under product distributions and because it behaves well under mixtures, making it easier to control the indistinguishability of P0^n and P1^n as the sample size grows. These divergences measure the geometry of the observation experiment, not the geometry of the parameter space. The art of applying the method is to balance two opposing requirements: the two worlds must be far apart in the answer space to force a large loss, but close in the observation space so that the sample cannot reliably reveal which world generated it.

This two-point approach is most powerful when the hardest part of the model class can already be witnessed by a single pair of distributions. It is the minimal lower-bound certificate: here are two admissible worlds, they demand different answers, and the data do not contain enough information to tell them apart. Because the argument only uses the existence of these two worlds and the optimal testing error between them, the bound applies to every measurable decision rule, including estimators that have not yet been invented. That makes the result more fundamental than analyzing a specific algorithm, which can only diagnose what that particular procedure does with the data.

The method also has natural limitations. When the true difficulty of the problem is many-way ambiguity, for instance because the minimax risk is driven by a combinatorial number of alternatives or by high-dimensional structure, a two-point argument may be loose. In those settings, packing methods such as Fano's inequality or Assouad's lemma can capture the dimension dependence more sharply. Nevertheless, Le Cam's two-point bound remains the cleanest way to isolate the basic information bottleneck, and it often gives the correct rate up to constants in problems such as normal mean estimation, nonparametric density estimation, functional estimation, and privacy-constrained inference.

To make the discussion concrete, consider the classical normal mean problem. Suppose under P0 the observations are independent N(0, 1) and under P1 they are independent N(delta, 1), with delta chosen proportional to 1 over the square root of n. Then the sample mean has variance 1/n, so the natural estimation rate is 1/n in squared error. Choosing delta = c / sqrt(n) for a small constant c makes the parameter separation 2s = c / sqrt(n), hence s = c / (2 sqrt(n)). The n-sample total variation between the two product normals is 2 Phi(c / 2) - 1, where Phi is the standard normal cumulative distribution function. For c = 1 this is about 0.38, so 1 - TV is about 0.62. The Le Cam bound then says the minimax squared error is at least a constant divided by n, which matches the rate achieved by the sample mean up to a constant factor. The following small Python script computes these quantities explicitly and verifies the numerical relationship between the testing error and the resulting risk lower bound.

```python
import math
import json

def phi_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def lecam_normal_mean(n=100, sigma=1.0, c=1.0):
    delta = c / math.sqrt(n)
    theta0, theta1 = 0.0, delta
    s = delta / 2.0
    # Exact total variation between the n-sample product normals
    tv = 2.0 * phi_cdf(delta * math.sqrt(n) / (2.0 * sigma)) - 1.0
    lecam_bound = (s * s / 2.0) * (1.0 - tv)
    # Equal-prior Bayes error of the optimal test based on the sample mean
    err0 = 1.0 - phi_cdf((delta / 2.0) * math.sqrt(n) / sigma)
    err1 = phi_cdf(((delta / 2.0) - delta) * math.sqrt(n) / sigma)
    bayes_error = 0.5 * (err0 + err1)
    test_based_lb = s * s * bayes_error
    return {
        "n": n,
        "sigma": sigma,
        "c": c,
        "separation_2s": delta,
        "radius_s": s,
        "TV": tv,
        "LeCam_squared_error_bound": lecam_bound,
        "bayes_test_error": bayes_error,
        "test_based_risk_lb": test_based_lb,
    }

if __name__ == "__main__":
    print(json.dumps(lecam_normal_mean(), indent=2))
```

In summary, Le Cam's two-point method reduces the search for a minimax lower bound to the construction of two distributions that are well separated in the parameter space yet statistically close in the observation space. By linking estimator accuracy to binary testing accuracy and invoking the exact optimal testing error between two laws, the method produces a universal lower bound that holds for every measurable estimator. It is the simplest certificate of statistical impossibility and remains a standard tool for proving that certain estimation rates cannot be improved.
