Let me start from what actually hurts. I am fine-tuning a language-model policy with policy-gradient RL, and I want to keep it from wandering too far from the frozen reference it started as. The thing I want to keep small is the Kullback-Leibler divergence between the two next-token distributions, KL[p_theta || p_ref] = sum_x p_theta(x) log(p_theta(x)/p_ref(x)), summed over the whole vocabulary at every position. And immediately I am stuck, because I cannot compute that sum. The vocabulary is tens of thousands of tokens; this has to be evaluated for every token of every sampled response on every optimizer step; and worse, in the training loop I only kept the log-probability of the token that was actually sampled, not the full distribution over the vocabulary. I have, per token, two scalars: log p_theta(x_t) for the token x_t my policy emitted, and log p_ref(x_t) for that same token under the reference. From those two numbers per token I have to manufacture something that behaves like the KL penalty. So the real question is not "what is KL" in the abstract; it is "how do I estimate a support-wide divergence when all I have is one sampled token and its two log-probs."

The way out is sitting in the definition itself if I read it as an expectation rather than as a sum I must enumerate. The tokens x_t were not arbitrary; they were sampled from the policy p_theta. So the vocabulary sum is an average over draws from p_theta: KL[p_theta || p_ref] = sum_x p_theta(x) log(p_theta(x)/p_ref(x)) = E_{x ~ p_theta}[log p_theta(x) - log p_ref(x)]. The weighting p_theta(x) in the sum is exactly the sampling distribution. That reframes the problem from "compute an intractable sum" to "estimate an expectation from samples." For any function g, E_{x ~ p_theta}[g(x)] is approximated by the sample average (1/N) sum_i g(x_i) over draws x_i ~ p_theta, and that average is unbiased, with error shrinking like sqrt(Var[g]/N). Here the function inside the expectation is g(x) = log p_theta(x) - log p_ref(x), and the draws are precisely the response tokens I already have. So the per-token quantity I should be summing is, just by reading off the integrand,

  log p_theta(x_t) - log p_ref(x_t),

i.e. logprob minus ref_logprob at each token, and the masked average of that over response tokens is a Monte-Carlo estimate of the KL. Nothing clever yet; I have only recognized that the thing I want is an expectation and written down the one-sample integrand of it. But let me make sure I believe the unbiasedness, because that is the whole reason this bare estimator is worth considering. The expectation of this per-token quantity over x ~ p_theta is, term by term, E_{x ~ p_theta}[log p_theta(x) - log p_ref(x)], which is exactly KL[p_theta || p_ref]. Its mean is the target with zero bias by construction, not approximately and not just to leading order.

Now let me stress-test it before I get comfortable, because "unbiased" is only half of "good." The other half is variance. Look at a single sample of log p_theta(x_t) - log p_ref(x_t). KL is a divergence, so it is nonnegative after averaging over all tokens, and yet this per-sample quantity is not. For a token the policy made less likely than the reference, log p_theta < log p_ref and the term is negative. So I am estimating a nonnegative quantity with signed samples whose average lands on the nonnegative KL. That is a warning sign: the estimate is relying on cancellation, and cancellation is where variance hides.

But "negative half the time" is too glib an explanation, and I do not want to fool myself with a slogan. Let me think about why the spread is actually large, not just that it is signed. The quantity is a log-ratio, log(p_theta/p_ref). Consider a token where the reference is confident and the policy is not: p_ref(x) is modest, p_theta(x) is small, and log(p_theta/p_ref) is a large negative number. Consider the reverse and it is a large positive number. Whenever one of the distributions assigns a tiny probability to the sampled token relative to the other, the log-ratio blows up in magnitude, positive or negative, and a few such tokens can dominate the finite-sample average. The sign flip is a symptom; the long-tailed log-ratio is the real source of noisy samples.

And the place this bites hardest is exactly the regime I am training in. The whole reason I have a KL penalty is to keep the policy close to the reference, so most of the time the true per-token KL is small. When the target quantity is tiny, a fixed absolute scatter in the estimate is a large relative error. A simple Gaussian sanity check makes this concrete: if two unit-variance Gaussians differ in mean by 0.1, the true KL is 0.005, and the naive log-ratio estimator has standard deviation about twenty times that true KL. That is already huge relative noise in the near-on-policy regime. Unbiased, yes; but each sample is mostly noise relative to the small signal. Wall: if I use this and nothing else, the KL value I add to the loss is going to be noisy, and noisy loss terms become noisy updates and diagnostics.

So can I keep the unbiasedness and reduce the variance? The standard tool for "reduce variance without adding bias" is a control variate. If I have an unbiased estimator m and any quantity t whose mean I know exactly, say E[t] = tau, then m + c(t - tau) is unbiased for every coefficient c, because the added piece has mean zero. The variance-minimizing c is -Cov(m,t)/Var(t), shrinking the variance according to the squared correlation. The catch is the known-mean quantity. What can I compute from the two log-probs of a sampled token whose mean is known for free? Write the ratio r = p_ref(x)/p_theta(x). Then E_{x ~ p_theta}[r] = sum_x p_theta(x) p_ref(x)/p_theta(x) = sum_x p_ref(x) = 1, so E_{x ~ p_theta}[r - 1] = 0. There it is: r - 1 is computable from the sampled log-probs as exp(ref_logprob - logprob) - 1, and its expectation is exactly zero no matter what the two distributions are. I can add lambda(r - 1) to the bare log-ratio estimator and unbiasedness survives for any lambda. The variance-optimal lambda depends on p_theta and p_ref and is not something I want to estimate inside the training loop, but lambda = 1 has a separate attraction: since log x <= x - 1, -log r + (r - 1) >= 0. That gives the always-nonnegative control-variate form (r - 1) - log r.

So I can do better than the bare log-ratio on variance. But hold on: I have changed the object in two ways I need to account for. First, the control-variate version forms r by exponentiating ref_logprob - logprob, so outlier tokens need numerical guarding before and after exp. Second, this penalty is not only logged as a scalar diagnostic; in the actor KL-loss path it is differentiated with respect to the policy. I have to ask not only whether the sampled value is unbiased for KL, but whether the gradient obtained by backpropagating through the sampled tensor is the true gradient of KL.

Let me compute that for the bare log-ratio. My per-token estimator is k1 = log p_theta(x) - log p_ref(x). When autodiff differentiates this with respect to theta, it treats the sampled token x as a fixed input, and p_ref does not depend on theta, so grad_theta k1 = grad_theta log p_theta(x) = s_theta(x), the score function at that token. Now take the expectation of that direct gradient over x ~ p_theta: E[grad_theta k1] = E[s_theta(x)] = sum_x p_theta(x) grad_theta log p_theta(x) = grad_theta sum_x p_theta(x) = grad_theta 1 = 0. The value is an exact unbiased estimator of KL, but the expected direct autodiff gradient of that sampled value is zero. As a standalone differentiable KL-loss term, this is not a directed pull toward the reference.

That looks contradictory only if I forget that the sampling distribution itself depends on theta. The true gradient of the expectation is grad_theta E_{x ~ p_theta}[log p_theta(x) - log p_ref(x)]. Differentiating the integrand gives E[s_theta]. Differentiating the measure gives the score-function term: grad_theta sum_x p_theta(x) f(x) = sum_x p_theta(x) s_theta(x) f(x) = E[s_theta f]. With f = k1, the true gradient is

  grad_theta KL[p_theta || p_ref] = E_{x ~ p_theta}[s_theta * k1] + E_{x ~ p_theta}[s_theta] = E_{x ~ p_theta}[s_theta * (k1 + 1)] = E_{x ~ p_theta}[s_theta * k1],

where E[s_theta] = 0 drops the +1. So the true KL gradient is the score weighted by the log-ratio. Naive autodiff through the sampled k1 value captures only the integrand-derivative term E[s_theta], which vanishes in expectation, and misses the score-function term from the theta-dependent sampling distribution. For k1, value-unbiased and gradient-correct are different properties.

This same calculation explains why the squared estimator has the gradient k1 lacks. Take 0.5 * (log p_theta - log p_ref)^2 = 0.5 * k1^2 and differentiate the sampled tensor: grad_theta(0.5 * k1^2) = k1 * grad_theta k1 = k1 * s_theta. That per-sample gradient term is exactly the score-weighted term whose expectation is the true KL gradient. So the squared estimator has biased value but the correct KL-gradient estimator, while the bare log-ratio has exact value but a mean-zero direct gradient. That asymmetry is why a straight-through construction exists at all: use one estimator for the forward value and substitute the squared estimator's backward pass with backward - backward.detach() + forward.detach().

Now I can place the bare log-ratio honestly. It is the simplest possible value estimator: one subtraction, no exp, no clamp, and E[k1] = KL[p_theta || p_ref] exactly. Its weaknesses are also clear: high variance, especially when KL is small, and a direct autodiff gradient whose expectation is zero rather than the true KL gradient. The squared estimator, the control-variate estimator, and the straight-through variants each repair one of those weaknesses at a cost: value bias, numerical guards around an exponential, or extra forward/backward machinery.

So the role of the bare log-ratio is the unadorned baseline. It gives the exact Monte-Carlo integrand of forward KL and nothing else. If I use it inside a differentiable KL-loss path, I must remember that the finite-batch gradient is noise around a zero-mean direct gradient, not an unbiased gradient of the expected KL. If I need low-variance values I reach for the control-variate form; if I need the KL gradient from a differentiable penalty I reach for the squared backward term or a straight-through variant. Here I keep the baseline bare, because the point is to preserve the exact sampled KL value with the least code and no numerical guard.

The implementation should therefore land as the branch that returns the sampled log-ratio inside the existing dispatcher. Same shape as the inputs, no detach, no exp in this branch; the other branches stay separate because they answer different variance or backward-pass needs:

```python
import torch


def kl_penalty_forward(
    logprob: torch.FloatTensor,
    ref_logprob: torch.FloatTensor,
    kl_penalty: str,
) -> torch.FloatTensor:
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        raise NotImplementedError

    raise NotImplementedError
```

The causal chain is: I need a per-token proxy for KL[p_theta || p_ref] but cannot enumerate the vocabulary, so I read the KL as an expectation over the policy's own samples; the one-sample integrand is logprob - ref_logprob, and its expectation is exactly KL. Stress-testing shows why that exactness is not enough: the signed log-ratio has high relative variance in the small-KL regime. The zero-mean identity E_{p_theta}[p_ref/p_theta - 1] = 0 explains the control-variate alternative, and the score-function derivation explains the loss-gradient mismatch: the true KL gradient is E[s_theta * k1], while direct autodiff through k1 has expectation E[s_theta] = 0. The bare branch keeps the value identity and accepts the variance and gradient limitations; that is precisely what makes it the naive baseline.
