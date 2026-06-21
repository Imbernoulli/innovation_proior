The object that controls everything when I fine-tune a language model on math with online RL is the advantage estimate: it is the single number that decides how hard each response and each token pushes the policy in the clipped PPO/GRPO surrogate. The clean way to get accurate advantages is a learned value function, but that critic is hurting me on two fronts at once. It is a second network the size of the policy, so it roughly doubles my memory and compute on one GPU; and in this setting the reward only lands on the EOS token, so the critic has almost no signal from which to learn an accurate per-token value. It is both the expensive part and the unreliable part. Deleting it is the obvious move, but the critic was never decoration — its job was to be the baseline I subtract from the return to cut the variance of the policy gradient, and removing it leaves a hole exactly where the baseline used to be. The critic-free methods fill that hole from the rewards of a few responses to the *same* prompt, and that small-group estimate is precisely where instability creeps in. RLOO baselines each response with the leave-one-out mean of its siblings, $A_i = r(o^{(i)}) - \frac{1}{k-1}\sum_{j\neq i} r(o^{(j)})$, which is genuinely unbiased but, with $k=4$ or $8$, is a noisy estimate of "the typical reward for this prompt," and it never puts the advantages from different prompts on a common scale. GRPO tries to fix the scale by also dividing by the group standard deviation, $A_i = (r_i - \mathrm{mean}(r))/(\mathrm{std}(r)+\varepsilon)$ broadcast to every token — but the local std is a brittle prompt-sized scale, and, as I will show, the local-std estimator is statistically biased at small group size because the centered numerator and the denominator are computed from the very same handful of rewards.

I propose REINFORCE++-baseline. The frame is plain full-sequence REINFORCE in the bandit view: with reward only at EOS there is no real per-token credit to assign, so I model the whole generation $o$ as a single action and the prompt $q$ as the state, maximizing $\mathbb{E}_{o\sim\pi_\theta}[R(o,q)]$ with the score-function gradient $\mathbb{E}_{o\sim\pi_\theta}[R(o,q)\,\nabla_\theta \log\pi_\theta(o\mid q)]$. With a single terminal reward the natural discount is $\gamma=1$ — there is nothing to discount, the return of the action *is* the terminal reward — and this is exactly what GAE collapses to at $\gamma=\lambda=1$ with no critic, where the advantage is just the centered full return. The variance-reduction fact I lean on is that for any baseline $b$ independent of the sampled action, $\mathbb{E}[(R-b)\nabla\log\pi]=\mathbb{E}[R\nabla\log\pi]$ because $\mathbb{E}[b\,\nabla\log\pi]=b\,\nabla\!\int\pi=0$, so I am free to pick any action-independent $b$ and want one close to $R$ on average. The method is a two-step normalization that fills the baseline hole: first subtract the per-prompt group mean as a local centering step, then whiten over the entire global batch of valid response tokens.

The reason I split the normalization this way — keep the centering, throw away the local scale — is a bias result I can prove rather than merely suspect. Model a group's rewards as $r_i=\theta+\varepsilon_i$ with $\varepsilon_i\sim\mathcal N(0,\sigma^2)$ i.i.d., $i=1\dots N$; what I would want the advantage to recover is $\varepsilon_i$. GRPO's estimator is $A_i=(\varepsilon_i-\bar\varepsilon)/D$ with $\bar\varepsilon=\frac1N\sum_j\varepsilon_j$ and $D=\sqrt{\frac1N\sum_j(\varepsilon_j-\bar\varepsilon)^2}$. The numerator behaves: conditioning on $\varepsilon_i$, the other residuals stay zero-mean, so $\mathbb{E}[\varepsilon_i-\bar\varepsilon\mid\varepsilon_i]=(1-\tfrac1N)\varepsilon_i$, a harmless known shrinkage if the denominator were constant. But the denominator is not constant given $\varepsilon_i$. Writing $D^2=\frac1N\sum_j\varepsilon_j^2-\bar\varepsilon^2$ and taking the conditional expectation gives

$$\mathbb{E}[D^2\mid\varepsilon_i]=\frac{(N-1)^2}{N^2}\sigma^2+\frac{N-1}{N^2}\varepsilon_i^2 \;=:\; \alpha+\beta\varepsilon_i^2,\qquad \beta=\frac{N-1}{N^2}>0.$$

The conditional second moment of the denominator *grows with* $\varepsilon_i^2$: a sample with large $|\varepsilon_i|$ systematically inflates the very spread it is then divided by. Taylor-expanding $f(x)=x^{-1/2}$ about $\mu=\mathbb{E}[D^2\mid\varepsilon_i]$,

$$f(x)=\mu^{-1/2}-\tfrac12\mu^{-3/2}(x-\mu)+\tfrac38\mu^{-5/2}(x-\mu)^2+O((x-\mu)^3),$$

the linear term dies under the conditional expectation, leaving $\mathbb{E}[1/D\mid\varepsilon_i]\approx\mu^{-1/2}+\tfrac38\,\mathrm{Var}(D^2\mid\varepsilon_i)\,\mu^{-5/2}$, whose leading scale $\mu^{-1/2}=(\alpha+\beta\varepsilon_i^2)^{-1/2}$ already varies with $\varepsilon_i^2$. I make this rigorous with a boundedness argument that avoids any false symmetry shortcut: set $z_j=\varepsilon_j-\bar\varepsilon$, so $\sum_j z_j=0$; then given $z_i$ the other residuals sum to $-z_i$, so $\sum_{j\neq i}z_j^2\ge z_i^2/(N-1)$, hence $D^2=\frac1N\sum_j z_j^2\ge z_i^2/(N-1)$ and $|A_i|\le\sqrt{N-1}$. The normalized local-std advantage is bounded for every finite $N$, while $\varepsilon_i$ is Gaussian and unbounded, so the conditional expectation cannot equal $\varepsilon_i$ as a function — the estimator is biased, and large advantages get compressed precisely because the sample that creates a large centered numerator also helps create the denominator that divides it. Crucially, as $N\to\infty$, $\beta\to0$ so $D\to\sigma$ and the $(1-\tfrac1N)$ factor $\to1$: the bias is a disease of small $N$, and the cure is more samples in whatever pool I standardize over.

I cannot grow the per-prompt group without paying $k\times$ the rollout cost, but I do have a large pool already in hand: the global batch of hundreds to a thousand-plus responses across many prompts in one optimization step. So I standardize the *scale* over that batch rather than per group. By the same limit, batch statistics computed over $N_{\text{global}}\approx1024$ converge to stable constants, any single sample has little leverage, and one tied or uninformative prompt can no longer set the scale for itself. Standardizing advantages over a pool — subtracting the mean and dividing by the std (whitening) before they enter the loss — is itself a standard PPO implementation knob (Andrychowicz et al. 2020); what is new is choosing the pool to be the global batch, because the *size* of the pool is exactly what controls the bias and the brittleness. I still keep the one good thing the group gives me: subtracting the per-prompt group mean encodes "this response beat the typical response *to this prompt*," which is more meaningful than raw reward, and it reshapes rewards so that whether the verifier emits $0/1$ or $-1/1$ the centered advantages land near zero — robustness to reward scale for free. The group mean includes the sample itself, so it is not the leave-one-out unbiased baseline; it carries the same harmless $(1-\tfrac1N)$ shrinkage I accounted for above. It is the local *std* — the biased, brittle part — that I discard. Singletons need a fallback: a group of size one has no siblings to average, and subtracting the sample from itself would zero its signal, so I set the group mean to $0$ for singletons and let the global whitening center them against the rest of the batch.

Two details make this token-level and consistent. The actor loss wants an advantage at every response position, so I broadcast the centered per-sequence scalar to every valid token. Then I whiten over the pool of *tokens*, not the pool of sequences: a 600-token response contributes 600 entries to the batch mean and std and a 50-token response contributes 50, so the statistics are length-weighted — the consistent choice, since I am standardizing exactly the per-token quantities that enter the gradient over exactly the set of tokens the loss will touch. The variance floor of $10^{-8}$ inside `masked_whiten` now sits under a batch statistic, so it is a numerical guard rather than the source of the scale. For the separate KL loss in the reasoning-training variant I add it as a loss term and must pick the estimator by its *gradient*, not its scalar value. Sampling from $\pi_\theta$ constrains the reverse KL, whose gradient is $\mathbb{E}_{y\sim\pi_\theta}[\ell(y)\nabla_\theta\log\pi_\theta(y)]$ with $\ell=\log\pi_\theta-\log\pi_{\text{ref}}$. The $k_1=\ell$ estimator is scalar-unbiased but differentiates to $\nabla\log\pi_\theta$, missing the log-ratio multiplier — fine inside the reward, wrong as a standalone loss. The $k_3=e^{-\ell}-1+\ell$ estimator is also scalar-unbiased, but $\nabla k_3=(1-e^{-\ell})\nabla\log\pi_\theta$ leaves the forward-KL gradient $-\mathbb{E}_{y\sim\pi_\theta}[(\pi_{\text{ref}}/\pi_\theta)\nabla\log\pi_\theta]$ after the score term vanishes, whose coefficient explodes when $\pi_\theta(y)$ is tiny. Only $k_2=\tfrac12\ell^2$ differentiates to exactly $\ell\,\nabla\log\pi_\theta$ — the reverse-KL gradient, with no importance ratio and no exponential to overflow — so the full objective is $L=L_{\text{PPO}}(A^{\text{norm}})-\lambda\,J_{k_2\text{ as loss}}(\theta)$. Everything else is PPO unchanged: the critic deleted, GAE taken at $\gamma=\lambda=1$ so the advantage is the centered full return, and the value baseline replaced by group-mean-then-global-whiten.

```python
from collections import defaultdict
from typing import Optional

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,         # (bs, response_length)
    index: torch.Tensor,                 # (bs,) prompt group id
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE++-baseline: per-prompt group-mean baseline, then global token-level whitening."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)              # outcome scalar per response

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)         # no local baseline for a singleton group
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]     # step 1: group-mean centering

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask  # broadcast to tokens
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask       # step 2: global whiten

    return scores, scores                                 # API returns advantages and return tensor
```
