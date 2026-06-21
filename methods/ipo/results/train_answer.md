We have a fixed, offline dataset of pairwise human preferences $(x, y_w, y_l)$ — for a context $x$, a rater said the generation $y_w$ beats $y_l$ — and we want to turn it into a better policy $\pi(y\mid x)$ that produces preferred generations while staying near a reference $\pi_\text{ref}$, with a single coefficient that genuinely controls *how near*. Cast cleanly, this is an offline contextual bandit, and the prevailing route fits a scalar Bradley–Terry reward $r$ via $p(y \succ y') = \sigma(r(y) - r(y'))$ and then maximizes the KL-regularized objective $J(\pi) = \mathbb{E}_{y\sim\pi}[r(y)] - \tau\,\mathrm{KL}(\pi \,\|\, \pi_\text{ref})$. There is a closed form I lean on throughout: for any per-action score $f$, the maximizer of $\mathbb{E}_\pi[f] - \tau\,\mathrm{KL}(\pi\|\pi_\text{ref})$ is the exponential tilt $\pi^*(y) \propto \pi_\text{ref}(y)\exp(\tau^{-1} f(y))$, unique whenever $\pi_\text{ref}$ has full support — this follows from the identity that the objective divided by $\tau$ equals $-\mathrm{KL}(\pi\|\pi^*) + \log Z$, a $\pi$-independent log-normalizer minus a KL that is maximized only at $\pi=\pi^*$. RLHF maximizes this with PPO against a learned reward; DPO (Rafailov et al. 2023) inverts the same optimum, $r(x,y) = \tau\log(\pi(y\mid x)/\pi_\text{ref}(y\mid x)) + \tau\log Z(x)$, so the normalizer cancels inside the Bradley–Terry difference and leaves a reward-free supervised loss $L_\text{DPO} = -\mathbb{E}_D[\log\sigma(\tau\log\tfrac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \tau\log\tfrac{\pi(y_l)}{\pi_\text{ref}(y_l)})]$.

The failure I keep hitting is what these do on a deterministic preference. Take $p^*(y \succ y') = 1$. Bradley–Terry can only represent this by sending the reward gap $r(y) - r(y') \to +\infty$; feed that into the optimal tilt and $\pi^*(y')/\pi^*(y) = (\pi_\text{ref}(y')/\pi_\text{ref}(y))\exp(\tau^{-1}(r(y') - r(y))) \to 0$, so $\pi^*(y') = 0$ *for every value of $\tau$*. I can crank $\tau$ to a million, demand the policy barely leave $\pi_\text{ref}$, and the optimum still annihilates $y'$: the KL term, the one thing meant to keep me near the reference, has quietly stopped binding. Worse, this is not a corner case once I admit I only see finite samples. A perfectly soft true preference $p^*(y\succ y') = 0.8$ can give an empirical estimate of exactly $1$ from two of two; the empirical optimum then zeroes $y'$ on a fluke. For language models, where actions are sequences and contexts are prompts, almost every pair is observed once or never, so empirical preferences land in $\{0,1\}$ constantly — the pathology is the typical case. RLHF dodges it only by accident: you cannot train a network to output $+\infty$ and people regularize the reward model on top, so the reward stays underfit and that finite, underfit gap is what keeps the policy regularized. DPO folded the reward away and fits only the policy through an *unbounded* logit inside $\log\sigma$, so it inherits the pathology in full and threw away the accidental shield.

The way out becomes clear once I see that both RLHF and DPO are members of one family. Pick a nondecreasing $\Psi:[0,1]\to\mathbb{R}$ and maximize $J(\pi) = \mathbb{E}_{y\sim\pi,\,y'\sim\mu}[\Psi(p^*(y\succ y'))] - \tau\,\mathrm{KL}(\pi\|\pi_\text{ref})$. Take the Bradley–Terry logit $\Psi(q) = \log(q/(1-q))$ and suppose $p^*(y\succ y') = \sigma(r(y)-r(y'))$; then $\mathbb{E}_{y'\sim\mu}[\Psi(p^*(y\succ y'))] = \mathbb{E}_{y'\sim\mu}[r(y)-r(y')] = r(y) - \text{const}$, which is the reward up to an additive constant, and the tilt only cares about scores up to additive constants — so the logit choice *is* RLHF and DPO. That pins the disease to a property of $\Psi$: the logit is **unbounded**, $\Psi(q)\to+\infty$ as $q\to 1$, which is exactly what let a deterministic preference overwhelm the fixed $\tau\,\mathrm{KL}$ term. A *bounded* score could never overwhelm the regularizer, because a single comparison contributes only a finite amount and $\tau$ can always be set to dominate it.

I propose IPO — Identity Preference Optimisation — the $\Psi = \text{identity}$ member of that family, the simplest bounded monotone choice. With $\Psi(q) = q$ the per-action score becomes the total preference of $y$ against the behavior distribution, $g(y) = \mathbb{E}_{y'\sim\mu}[p^*(y\succ y')] =: p^*(y\succ\mu) \in [0,1]$, and the objective is direct regularized maximization $\max_\pi\, p^*(\pi\succ\mu) - \tau\,\mathrm{KL}(\pi\|\pi_\text{ref})$ — no logit, no Elo, no Bradley–Terry assumption that pairwise preferences reduce to pointwise rewards. By the tilt formula its optimum is $\pi^*(y)\propto\pi_\text{ref}(y)\exp(\tau^{-1} p^*(y\succ\mu))$, and because the exponent is bounded in $[0,1]$, no matter how deterministic any individual preference is the exponent cannot run off to infinity and $\tau$ keeps biting.

To get this as an offline loss with no RL, I turn the analytic optimum into root-finding. Taking the ratio of the tilt for two actions kills the normalizer, and defining the reference-corrected log-ratio $h_\pi(y,y') = \log(\pi(y)\pi_\text{ref}(y')/(\pi(y')\pi_\text{ref}(y)))$, the optimum satisfies one scalar equation per ordered pair, $h^*(y,y') = \tau^{-1}(g(y) - g(y'))$. Here I peel away from the Bradley–Terry likelihood Rafailov plugged into — with $\Psi = I$ there is no logit to invert — and instead fold these per-pair equations into a single squared-residual objective
$$L(\pi) = \mathbb{E}_{y,y'\sim\mu}\!\left[\left( h_\pi(y,y') - \frac{p^*(y\succ\mu) - p^*(y'\succ\mu)}{\tau}\right)^2\right].$$
It is an expectation of squares, so $L\ge 0$, and the residual vanishes at $\pi^*$, so $\pi^*$ is a global minimizer. To rule out spurious minima, parametrize $\pi$ by logits $s$ with $\pi_s(y) = e^{s(y)}/\sum_{y'} e^{s(y')}$; then $L$ is quadratic in $s$, its pure-quadratic part is $\sum_{y,y'} \mu(y)\mu(y')(s(y)-s(y'))^2$, a positive-*semidefinite* form, so $L$ is convex and every local minimizer is global. The only flat direction is the all-ones shift $s\to s+\lambda\mathbf{1}$, which leaves every difference $s(y)-s(y')$ fixed — and that direction is policy-invariant, since the softmax quotients out a constant added to all logits. So the minimizing *policy* is unique, provided $\mathrm{Supp}(\mu) = \mathrm{Supp}(\pi_\text{ref})$; if $\pi$ ranges over a larger support than $\mu$, pairs the loss never samples stay unconstrained and uniqueness genuinely fails.

The remaining obstacle is that $p^*(y\succ\mu)$ is unobservable — I see only Bernoulli labels $I(y,y')$ with mean $p^*(y\succ y')$. Replace the unknown gap with the raw label and consider the sampled loss $\mathbb{E}_{y,y'\sim\mu}[(h_\pi(y,y') - \tau^{-1} I(y,y'))^2]$. Term by term this is *not* the true loss, since conditioning on a fixed pair gives the single pairwise preference $p^*(y\succ y')$ rather than the total-preference difference. But the equality holds in expectation up to a $\pi$-independent constant, which is all the argmin needs. Since $I^2 = I$ for a Bernoulli and the gaps are bounded, the only $\pi$-dependent difference is the cross term, so it suffices that $\mathbb{E}[h_\pi I] = \mathbb{E}[h_\pi(p_y - p_{y'})]$ with $p_y := p^*(y\succ\mu)$. The lever is that $h_\pi$ is **additive and antisymmetric**: writing $a_y = \log\pi(y) - \log\pi_\text{ref}(y)$, we have $h_\pi(y,y') = a_y - a_{y'}$. Using iid $y,y'$ and $\mathbb{E}_\mu[p_y] = 1/2$ (from $p^*(y\succ y') + p^*(y'\succ y) = 1$ and exchangeability), the right side expands to $\mathbb{E}_\mu[(2p_y - 1)a_y]$. The left side splits across $y$ and $y'$, and partner-averaging the label recovers the total preference exactly — $\mathbb{E}_{y'}[I(y,y')\mid y] = p^*(y\succ\mu) = p_y$ and $\mathbb{E}_y[I(y,y')\mid y'] = 1 - p_{y'}$ — giving $\mathbb{E}_\mu[(2p_y-1)a_y]$ as well. The two sides agree: the partner-averaging is what turns a single pairwise label into the total-preference gap, exploiting precisely the additive/antisymmetric structure.

Now make it empirical. Each recorded comparison $(y_w, y_l)$ furnishes both ordered terms $(y,y',I) = (y_w,y_l,1)$ and the swapped $(y_l,y_w,0)$; using both halves cuts variance for free. Averaging and using antisymmetry $h_\pi(y_l,y_w) = -h_\pi(y_w,y_l)$, with $H := h_\pi(y_w,y_l)$, the bracket is $(H - \tau^{-1})^2 + H^2 = 2H^2 - 2\tau^{-1}H + \tau^{-2} = 2(H - \tau^{-1}/2)^2 + \tau^{-2}/2$; halving and dropping the $\pi$-independent constant collapses everything to a single, strikingly simple squared term
$$L_\text{IPO}(\pi) = \mathbb{E}_{(y_w,y_l)\sim D}\!\left[\left( h_\pi(y_w, y_l) - \frac{1}{2\tau}\right)^2\right], \qquad h_\pi(y_w,y_l) = \log\frac{\pi(y_w)}{\pi(y_l)} - \log\frac{\pi_\text{ref}(y_w)}{\pi_\text{ref}(y_l)}.$$
This regresses the gap between the policy's winner-over-loser log-ratio and the reference's onto *one fixed finite target* $\tau^{-1}/2$, the same for every pair. Unlike DPO's $\log\sigma$, there is no saturating sigmoid that keeps paying out as the log-ratio grows — there is a finite value the policy wants to *sit at*, and once it separates winner from loser by $\tau^{-1}/2$ more than the reference does, the gradient vanishes. Weaken regularization (smaller $\tau$, larger target) and the allowed separation grows; strengthen it (large $\tau$, target $\to 0$) and the policy is pulled to match $\pi_\text{ref}$'s own separation. The coefficient does exactly what it advertises, even on deterministic data, because the target is a finite number rather than a logit running to infinity. On the minimal instance — two actions, $p^*(y_1\succ y_2)=1$, uniform $\pi_\text{ref}$ and $\mu$ — the total preferences are $p^*(y_1\succ\mu) = 3/4$ and $p^*(y_2\succ\mu) = 1/4$, so $\pi^*(y_1) = \sigma(\tau^{-1}/2)$: as $\tau\to\infty$ this returns the uniform $\pi_\text{ref}$, as $\tau\to 0$ it returns the deterministic policy, the whole continuum swept by $\tau$ — which the logit objective, frozen at $\pi(y_2)=0$ for all $\tau$, never could.

One decision the bandit abstraction did not force but the sequence setting does: for a language model the log-probability of a generation $y$ is the sum of per-token log-probs over the completion, so a raw-sum $h_\pi$ would scale with completion length and make the single target $\tau^{-1}/2$ mean different things at different lengths. The fix is to use the *average* per-token log-probability — divide each sequence log-prob by its completion length — so $h_\pi$ is on a per-token scale and the one target is comparable across pairs of any length. In code the harness exposes the regularization coefficient under the name `beta`, so `beta` plays the role of $\tau$ and the target is written `1/(2*beta)`; the per-example squared term is averaged over the batch to give the scalar to backpropagate. The loss drops straight into the existing harness's empty `preference_objective` slot:

```python
import torch


def preference_objective(policy_chosen_lp, policy_rejected_lp,
                         ref_chosen_lp, ref_rejected_lp,
                         chosen_len, rejected_len, beta):
    """IPO loss for one batch of preference pairs.

    Args are sequence-level summed log-probs of policy and reference on the
    chosen / rejected completions, the two completion lengths, and beta (= the
    KL-regularization coefficient tau; the regression target is 1/(2*beta)).
    """
    chosen_len = chosen_len.clamp_min(1).to(policy_chosen_lp.dtype)
    rejected_len = rejected_len.clamp_min(1).to(policy_rejected_lp.dtype)

    # per-token average log-probs: keeps h_pi on a per-token scale so the single
    # fixed target is comparable across completions of different lengths.
    pol_chosen   = policy_chosen_lp   / chosen_len
    pol_rejected = policy_rejected_lp / rejected_len
    ref_chosen   = ref_chosen_lp      / chosen_len
    ref_rejected = ref_rejected_lp    / rejected_len

    # reference-corrected log-ratio gap  h_pi(y_w, y_l) =
    #   [log pi(y_w) - log pi_ref(y_w)] - [log pi(y_l) - log pi_ref(y_l)]
    chosen_logratio   = pol_chosen   - ref_chosen
    rejected_logratio = pol_rejected - ref_rejected
    h_pi = chosen_logratio - rejected_logratio

    # regress the gap onto the single fixed target 1/(2*beta) = tau^{-1}/2;
    # bounded target keeps KL alive even for deterministic preferences.
    losses = (h_pi - 1.0 / (2.0 * beta)) ** 2
    return losses.mean()
```

In the shape it actually ships, a trainer length-averages the policy and reference sequence log-probs, forms the same averaged $h_\pi$, and squares the distance to `1/(2*beta)`, carrying implicit-reward signals $\beta\,(\log\pi - \log\pi_\text{ref})$ for logging only:

```python
import torch


class IPOTrainer:
    def __init__(self, policy_model, reference_model, beta=0.1, lr=5e-7):
        self.policy = policy_model
        self.ref = reference_model.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.beta = beta
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @staticmethod
    def avg_seq_logp(model, input_ids, completion_mask):
        logits = model(input_ids).logits[:, :-1, :].log_softmax(-1)
        labels = input_ids[:, 1:]
        per_token = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(2)
        mask = completion_mask[:, 1:]
        length = mask.sum(-1).clamp_min(1).to(per_token.dtype)
        return (per_token * mask).sum(-1) / length          # average log-prob

    def compute_preference_loss(self, pol_c, pol_r, ref_c, ref_r):
        h_pi = (pol_c - ref_c) - (pol_r - ref_r)            # reference-corrected gap
        losses = (h_pi - 1.0 / (2.0 * self.beta)) ** 2      # IPO: squared distance to fixed target
        chosen_rewards = self.beta * (pol_c - ref_c).detach()
        rejected_rewards = self.beta * (pol_r - ref_r).detach()
        return losses.mean(), chosen_rewards, rejected_rewards

    def train_step(self, batch):
        pol_c = self.avg_seq_logp(self.policy, batch["chosen_ids"],   batch["chosen_mask"])
        pol_r = self.avg_seq_logp(self.policy, batch["rejected_ids"], batch["rejected_mask"])
        with torch.no_grad():
            ref_c = self.avg_seq_logp(self.ref, batch["chosen_ids"],   batch["chosen_mask"])
            ref_r = self.avg_seq_logp(self.ref, batch["rejected_ids"], batch["rejected_mask"])
        loss, _, _ = self.compute_preference_loss(pol_c, pol_r, ref_c, ref_r)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss
```
