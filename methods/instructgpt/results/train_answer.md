We have a large decoder-only transformer pretrained to predict the next token on a giant slice of internet text, and it is genuinely good at that — but the objective it was trained on, maximizing $\sum_t \log p(x_t \mid x_{<t})$ over web text, is a proxy, and it is the wrong proxy for what people actually want when they use the model. The model continues text the way some webpage would; it does not do what a user is asking, helpfully, truthfully, and harmlessly. So it answers a slightly different question than the one posed, ignores explicit constraints, fabricates facts with confidence, or emits toxic output. The most uncomfortable part is that scale does not rescue us: a model with ten times the parameters is more fluent but just as misaligned. Capability and intent-following are decoupling, which means the fix cannot be "train a bigger model on the same objective" — we have to change *what we optimize for*, and the thing we care about ("is this output what the user wanted, and is it honest and harmless?") lives only in human judgment, with no formula to write down.

The obvious first move is supervised fine-tuning on demonstrations: hire people to write ideal responses to prompts and fine-tune with the ordinary next-token cross-entropy on those responses. This is necessary — it teaches the model the *shape* of the task, that an instruction calls for a direct on-task response rather than a document continuation — but it has a hard ceiling. Behavior cloning is capped by imitation: there is one demonstration per prompt, and for open-ended tasks there is a whole landscape of better and worse responses about which a single demonstration says nothing. The model learns to produce *a* plausible answer, never to discriminate better from worse, and can never exceed the demonstrator. Empirically the supervised loss even betrays its own irrelevance — validation cross-entropy bottoms out and overfits after one epoch while true downstream quality keeps climbing. Few-shot prompting of the untouched base model is more brittle still, putting the burden on the user to find a working prefix and leaving hallucination and toxicity untouched. Cross-task instruction tuning on academic NLP datasets (FLAN, T0) improves held-out academic tasks but aligns on the wrong distribution: those collections are dominated by tasks that are easy to auto-score, not the open-ended generation and brainstorming that real prompts consist of. And optimizing human judgment directly with RL hits a fatal throughput wall — RL needs millions of episodes and a human can judge only a few hundred outputs a day, five or six orders of magnitude too slow. The objective we actually care about is exactly the one we cannot query at the rate RL demands.

I propose InstructGPT: reinforcement learning from human feedback assembled as three stages, each built precisely to break the previous stage's ceiling. The first stage is supervised fine-tuning to enter instruction-following mode and, crucially, to provide the anchor policy $\pi^{\text{SFT}}$ that the later stages lean on. The second stage solves "imitation cannot discriminate" by learning to discriminate. Since humans cannot ride inside the RL loop, I query them *offline* once, collect their judgments into a dataset, and fit a model $r_\theta(x,y)$ of human judgment — a neural net I can then call millions of times for free. The key choice is *what* to ask the humans: not absolute 1–7 ratings (labelers anchor differently and drift), but *comparisons*, which people give far more reliably. For a prompt $x$ a labeler is shown $K$ responses at once ($K\in[4,9]$) and ranks them, yielding $\binom{K}{2}$ pairwise comparisons for the price of one labeling session — a large throughput multiplier over showing two at a time.

The modeling puzzle is to turn ordinal "A beats B" data into a loss on a real-valued $r_\theta$. If $y_w$ is preferred to $y_l$ I want $r_\theta(x,y_w) > r_\theta(x,y_l)$, with a larger gap the more often humans prefer $y_w$ — so the probability of preference should be an increasing function of the difference $r_\theta(x,y_w)-r_\theta(x,y_l)$. This is the classical paired-comparison setup. Give each response a latent strength $s=\exp(r)$ (exponential so it is positive and the algebra comes out linear in $r$); then the Bradley–Terry model says

$$P(y_w \succ y_l) = \frac{s_w}{s_w+s_l} = \frac{e^{r_w}}{e^{r_w}+e^{r_l}},$$

and dividing numerator and denominator by $e^{r_w}$ collapses this to

$$P(y_w \succ y_l) = \frac{1}{1+e^{-(r_w-r_l)}} = \sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big).$$

The reward *difference* is the log-odds of preference. Two consequences: the probability depends only on the difference, so $r$ is identifiable only up to an additive constant (I pin this gauge later by normalizing so demonstration responses have mean reward $0$), and fitting it by maximum likelihood is exactly logistic regression. The negative log-likelihood is $-\log\sigma(r_w-r_l)$, and the gradient confirms the sign is right: with $u=r_w-r_l$, $\frac{d}{du}[-\log\sigma(u)] = \sigma(u)-1$, so descent raises $r_w$ and lowers $r_l$, with magnitude $1-\sigma(u)$ that is large when the pair is currently mis-ranked and vanishes once it is ranked correctly — the loss self-throttles. Architecturally I do not learn to read English from scratch: the reward model is the SFT model with its unembedding replaced by a small linear head emitting one scalar, so the transformer does the comprehension and I learn only the readout. I deliberately keep the reward model at a fixed mid-sized scale (6B) used for policies of all sizes; the largest reward models are costly and unstable, and this one must later initialize the value function during RL where instability is poison.

One subtlety in consuming the ranked data: the naive thing is to flatten every $\binom{K}{2}$ pair into the dataset as an independent shuffled example, and it backfires badly. Each completion then appears in $K-1$ pairs, driving that many correlated gradient updates per epoch, and the reward model overfits in a single pass — memorizing the prompt's particular completions rather than learning preference. The fix is to treat all $\binom{K}{2}$ comparisons from one prompt as *one* batch element: run each of the $K$ completions through the reward model exactly once ($K$ forward passes, not $2\binom{K}{2}$), cache the scalars, and form all pairwise terms from the cache. This wins on compute and on generalization at once, because a prompt's correlated comparisons are consumed together in one coherent update. For a ranked prompt with pair set $\mathcal P(x)$,

$$\ell_x(\theta)=-\frac{1}{\binom{K}{2}}\sum_{(y_w,y_l)\in\mathcal P(x)}\log \sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big),\qquad \mathrm{loss}(\theta)=\mathbb E_{x\sim D_{\mathrm{rank}}}[\ell_x(\theta)],$$

trained for a single epoch (more and it overfits fast).

The third stage optimizes the policy against this cheap, fast reward. Structurally this is the simplest RL — a bandit: the state is the prompt $x$, the action is the entire response $y$, and the episode is one step (sample $y$, score it, done). I want to maximize $\mathbb{E}_x\,\mathbb{E}_{y\sim\pi_\phi}[r_\theta(x,y)]$, and the score-function estimator $\nabla_\phi\mathbb{E}[r]=\mathbb{E}[r(x,y)\nabla_\phi\log\pi_\phi(y\mid x)]$ says to push up the log-probability of high-reward responses. Vanilla REINFORCE fails for two reasons. First, it is high-variance and an unconstrained step can lurch a language-model policy into degenerate, repetitive text it never recovers from; I need a per-update trust region. Rather than a heavy second-order method I use PPO's cheap first-order one: form the importance ratio $\rho_t=\pi_\phi(a_t\mid s_t)/\pi_{\phi_{\text{old}}}(a_t\mid s_t)$ and maximize the *clipped* surrogate

$$\min\!\Big(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\,\hat A_t\Big).$$

Checking the cases shows why this works: when $\hat A_t>0$, raising $\rho_t$ helps only until $\rho_t=1+\epsilon$, past which the $\min$ selects the cap and removes the incentive to push further, while if $\rho_t$ is below the band the unclipped term still pushes the probability up; when $\hat A_t<0$ the mirror holds, with no incentive to suppress a token below $\rho_t=1-\epsilon$ but a penalty if it overshoots above the band. The $\min$ is pessimistic — it clips exactly the changes that would flatter the surrogate beyond what I trust. Since the code minimizes a loss, this appears as $\max(-\rho_t\hat A_t,\ -\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)$. I use $\epsilon=0.2$, advantages from generalized advantage estimation ($\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ with $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$, no discount, $\lambda=0.95$), a single inner epoch per batch, and a value function initialized from the reward model since it already maps text to a scalar.

The second reason vanilla RL fails is specific to optimizing a *learned* reward, and it is the central danger: $r_\theta$ is only an approximation, trustworthy only near the SFT distribution it was trained on. A relentless optimizer will find adversarial regions where $r_\theta$ reports a high score but a human would recoil — reward over-optimization — so the harder I push, the more the reported score decouples from real preference. Both failure modes point at the same remedy: keep the policy near $\pi^{\text{SFT}}$, which is exactly where the reward model is valid and the policy is well-behaved. I add a per-token KL penalty $\mathrm{KL}(\pi_\phi\,\|\,\pi^{\text{SFT}})$, estimated from the sampled sequence by its log-ratio $\log\pi_\phi(y\mid x)-\log\pi^{\text{SFT}}(y\mid x)$, folded straight into the reward the optimizer sees:

$$R(x,y)=r_\theta(x,y)-\beta\Big(\log\pi_\phi(y\mid x)-\log\pi^{\text{SFT}}(y\mid x)\Big),$$

with the RM score applied at the final token and the KL term as a per-token penalty along the way. This leashes the policy to where the reward is honest and, by penalizing collapse onto any single response, doubles as an entropy-like regularizer. The coefficient $\beta$ is the leash length: too small and the policy games the reward model, too large and it is pinned to SFT and cannot improve; empirically $\beta\approx 0.02$.

Running this surfaces one more problem I should have anticipated: after RL the model has *regressed* on standard public NLP benchmarks — an alignment tax, bad on its own terms and bad strategically. My first instinct, tighten the KL leash by cranking $\beta$ up a hundredfold, does *not* fix the regressions and tanks the reward. That is informative: the KL-to-SFT anchor is the wrong anchor, because the SFT model has itself drifted from the broad pretrained distribution where the lost capabilities actually live. So I point the regularizer there directly — mix the original pretraining objective back into the updates, anchoring to the exact distribution the capabilities came from. The full objective maximized over the RL policy is

$$\mathrm{objective}(\phi)=\mathbb{E}_{(x,y)\sim \pi^{\text{RL}}_\phi}\Big[r_\theta(x,y)-\beta\log\frac{\pi^{\text{RL}}_\phi(y\mid x)}{\pi^{\text{SFT}}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}\big[\log\pi^{\text{RL}}_\phi(x)\big].$$

In practice I accumulate the PPO gradient and the pretraining negative-log-likelihood gradient before stepping; minimizing $L_{\mathrm{PPO}}+\gamma L_{\mathrm{ptx}}$ matches maximizing the objective since $L_{\mathrm{ptx}}=-\mathbb{E}[\log\pi_\phi(x)]$. The coefficient $\gamma$ trades capability retention against alignment reward; I need it reasonably large, $\gamma\approx 27.8$ (on the order of $\gamma\gtrsim 20$ to counter the worst regressions), with enough pretraining data per update — eight pretraining examples per RL episode is the compromise between too little (the ptx loss creeps up) and too much (the update slows badly). Setting $\gamma=0$ recovers plain PPO; the version I want, **PPO-ptx = InstructGPT**, carries both forces in its gradient: preference optimization from PPO and capability retention from the pretraining likelihood. The whole pipeline assembled itself out of necessity — imitation cannot discriminate, so the reward model adds discrimination; humans cannot be queried online, so the reward model is queried instead; the reward model can be gamed, so the KL leash keeps optimization honest; and RL extracts a tax the KL leash cannot repay, so the pretraining mix anchors the capabilities.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)

def gather_token_logprobs(logits, input_ids):
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    labels = input_ids[:, 1:]
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def response_nll(model, input_ids, loss_mask):
    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    token_logp = gather_token_logprobs(logits, input_ids)
    return -masked_mean(token_logp, loss_mask[:, 1:])

def train_demonstration_model(model, demo_loader, optimizer):
    for input_ids, loss_mask in demo_loader:
        loss = response_nll(model, input_ids, loss_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class PreferenceScorer(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        last = attention_mask.long().sum(dim=1) - 1
        row = torch.arange(h.size(0), device=h.device)
        return self.score(h[row, last]).squeeze(-1)

def preference_loss(scores_chosen, scores_rejected, margin=None, pair_mask=None):
    diff = scores_chosen - scores_rejected
    if margin is not None:
        diff = diff - margin
    losses = -F.logsigmoid(diff)
    if pair_mask is not None:
        pair_mask = pair_mask.to(losses.dtype)
        pair_counts = pair_mask.sum(dim=-1).clamp_min(1)
        per_prompt = (losses * pair_mask).sum(dim=-1) / pair_counts
        valid_prompts = (pair_mask.sum(dim=-1) > 0).to(losses.dtype)
        return (per_prompt * valid_prompts).sum() / valid_prompts.sum().clamp_min(1)
    if losses.ndim > 1:
        return losses.mean(dim=-1).mean()
    return losses.mean()

def train_preference_scorer(scorer, comparison_loader, optimizer):
    # The loader can group all pairs from ranked prompts as [batch, pairs, tokens].
    def score_pair_batch(input_ids, attention_mask):
        if input_ids.dim() == 3:
            bsz, pairs, seqlen = input_ids.shape
            flat_scores = scorer(input_ids.reshape(bsz * pairs, seqlen),
                                 attention_mask.reshape(bsz * pairs, seqlen))
            return flat_scores.view(bsz, pairs)
        return scorer(input_ids, attention_mask)

    for batch in comparison_loader:
        chosen = score_pair_batch(batch.input_ids_chosen, batch.attention_mask_chosen)
        rejected = score_pair_batch(batch.input_ids_rejected, batch.attention_mask_rejected)
        loss = preference_loss(chosen, rejected,
                               margin=getattr(batch, "margin", None),
                               pair_mask=getattr(batch, "pair_mask", None))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def token_rewards(scores, logprobs, ref_logprobs, masks, kl_coef):
    # Response-only logprobs/ref_logprobs/masks (TRL convention).
    kls = logprobs - ref_logprobs
    non_score_rewards = -kl_coef * kls
    rewards = non_score_rewards.clone()
    for i, (score, mask) in enumerate(zip(scores, masks)):
        last = mask.nonzero(as_tuple=False)[-1].item()
        rewards[i, last] += score
    return rewards, non_score_rewards, kls

def masked_whiten(values, mask, shift_mean=True):
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    return whitened if shift_mean else whitened + mean

def estimate_advantages(values, rewards, mask, gamma=1.0, lam=0.95):
    values = values * mask
    rewards = rewards * mask
    lastgaelam = 0
    reversed_advantages = []
    for t in reversed(range(rewards.shape[-1])):
        nextvalues = values[:, t + 1] if t < rewards.shape[-1] - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        reversed_advantages.append(lastgaelam)
    advantages = torch.stack(reversed_advantages[::-1]).transpose(0, 1)
    returns = advantages + values
    advantages = masked_whiten(advantages, mask).detach()
    return values, advantages, returns

def clipped_policy_loss(old_logprobs, values, logits, vpreds, logprobs,
                        mask, advantages, returns, cliprange=0.2,
                        cliprange_value=0.2, vf_coef=0.1):
    vpred_clipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpred_clipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

    ratio = torch.exp(logprobs - old_logprobs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mask)
    return pg_loss + vf_coef * vf_loss

def train_policy(policy, value_model, scorer, reference_policy,
                 prompt_loader, retention_loader, optimizer,
                 kl_coef, retention_coef):
    for prompts, retention_batch in zip(prompt_loader, retention_loader):
        responses, old_logprobs = policy.generate_with_logprobs(prompts)
        model_inputs = pack(prompts, responses)
        response_mask = model_inputs.response_mask

        with torch.no_grad():
            ref_logprobs = reference_policy.logprobs(model_inputs)
            scores = scorer(model_inputs.input_ids, model_inputs.attention_mask)
            old_values = value_model(model_inputs)

        rewards, _, _ = token_rewards(scores, old_logprobs, ref_logprobs, response_mask, kl_coef)
        old_values, advantages, returns = estimate_advantages(old_values, rewards, response_mask)

        logits, logprobs, vpreds = policy.forward_with_values(model_inputs)
        ppo = clipped_policy_loss(old_logprobs.detach(), old_values.detach(), logits, vpreds,
                                  logprobs, response_mask, advantages, returns.detach())
        ptx = response_nll(policy, retention_batch.input_ids, retention_batch.loss_mask)

        optimizer.zero_grad()
        (ppo + retention_coef * ptx).backward()
        optimizer.step()
```
