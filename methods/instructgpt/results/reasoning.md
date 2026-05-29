OK, let me think this through from scratch. I have a pretrained language model — a big decoder-only transformer trained to predict the next token on a giant pile of internet text — and it is genuinely good at that. But when someone actually tries to *use* it, it disappoints in maddeningly consistent ways: I ask it to do a task and it answers a slightly different question, or it makes up a fact with total confidence, or it ignores a constraint I spelled out ("answer in two sentences"), or it produces something toxic. None of this is a bug in the code. It's doing exactly what it was trained to do — continue the text the way a random webpage might — and that is just not the same thing as "do what this person is asking, helpfully and without lying or causing harm."

So the real problem is an objective mismatch. The training objective, maximize $\sum_t \log p(x_t \mid x_{<t})$ over web text, is a *proxy*, and it's the wrong proxy for "follow the user's intent." And here's the part that bothers me most: I keep hoping scale will rescue me. It doesn't. A model with ten times the parameters is more capable but just as misaligned — it hallucinates and ignores instructions with the same enthusiasm, only more fluently. Capability and intent-following are coming apart. That tells me I can't fix this by training a bigger model on the same objective; I have to change *what I'm optimizing for*. The thing I actually care about — "is this output what the user wanted, and is it honest and harmless?" — lives in human judgment, and I have no formula for it.

Let me try the most obvious thing first. If I want the model to follow instructions, why not just show it how? Hire people to write high-quality responses to a bunch of prompts, then fine-tune the model to maximize the likelihood of those responses. Ordinary supervised learning — same next-token cross-entropy, just on demonstrations of good behavior instead of random web text. This is behavior cloning, and I should absolutely do it, because it teaches the model the *shape* of the thing: when you see an instruction, you produce a direct, on-task response in this register. Without it, the model doesn't even know it's supposed to be in "answer the user" mode rather than "continue the document" mode.

But I can already feel the ceiling. Behavior cloning is capped by imitation. For each prompt I have one demonstration, and for open-ended tasks — "write a short story where a bear goes to the beach," "brainstorm five ways to regain enthusiasm for my career" — there isn't *one* right answer; there's a whole landscape of better and worse responses, and a single demonstration carries no information about that landscape. The model learns to produce *a* plausible answer, never to tell a better one from a worse one, and it certainly can't learn to be *better* than the people writing the demonstrations. And empirically the supervised loss betrays me: validation loss bottoms out and starts overfitting after a single epoch, yet if I keep training, downstream quality keeps improving. That's a dead giveaway that validation cross-entropy is not the quantity I care about. The target I'm minimizing and the target I want have parted ways. Imitation gets me to the starting line, not past it.

What I really want is to optimize human preference *directly*. So picture the model as a policy $\pi$: it sees a prompt $x$, it samples a response $y$, and a human looks at $y$ and tells me how good it is. Use that as a reward, and do reinforcement learning — push up the probability of responses humans like. That's exactly the right objective; it's the thing I actually care about, not a proxy.

And immediately I hit the wall. RL is brutally sample-inefficient. To get a policy-gradient signal that isn't pure noise I need to evaluate enormous numbers of rollouts — hundreds of thousands, millions of responses. There is no way to put a human in that loop. A person can judge maybe a few hundred outputs a day; the RL algorithm wants to judge millions. The human is five or six orders of magnitude too slow. So "optimize human judgment directly online" is infeasible. I'm stuck — the right objective is the one I can't afford to query.

The human is expensive and slow, but I don't actually need the human *inside* the RL loop. What if I query humans *offline*, collect their judgments into a dataset, and then fit a *model* of human judgment — a function $r(x, y)$ that takes a prompt and a response and outputs a scalar predicting how much a human would like it? Then I optimize the policy against $r$, which is just a neural net: I can call it as many millions of times as I want, for free. The humans are queried once, to build $r$'s training set; the RL never touches a human again. That's the trade. Pay humans to teach a cheap, fast proxy for their own judgment, then let the optimizer hammer on the proxy. This is exactly the move that earlier work made for robots and Atari, where the "reward" was likewise something you could recognize but not write down.

So now I need to actually build this reward model. First question: what do I ask the humans? My instinct is to ask "rate this response 1 to 7." But absolute ratings are a mess — different labelers anchor differently, the same labeler drifts over a day, and "is this a 5 or a 6?" is genuinely hard. What people *are* reliable at is comparison: show them two responses to the same prompt and ask which is better. Relative judgment is far more consistent than absolute scoring. So I'll collect *comparisons*: for a prompt $x$, two responses $y_w$ and $y_l$, the human says $y_w$ is preferred.

Now I have a modeling puzzle: I want a *scalar* reward $r_\theta(x, y)$, but my data is *ordinal* — "this one beat that one." How do I turn a pile of "A beats B" into a loss on a real-valued function? Let me think about what it should mean for $r$ to be consistent with a comparison. If $y_w$ is preferred to $y_l$, I want $r_\theta(x, y_w) > r_\theta(x, y_l)$, and the *more* often humans prefer $y_w$, the bigger I want the gap to be. So the probability a human prefers $y_w$ should be some increasing function of the difference $r_\theta(x, y_w) - r_\theta(x, y_l)$.

This is exactly the classical paired-comparison setup. Suppose each response has a latent "strength," and the chance one beats another depends only on their strengths. If I write strength as $s = \exp(r)$ — exponential so it's positive, and so the algebra comes out linear in $r$ — then the natural model is

$$P(y_w \succ y_l) = \frac{s_w}{s_w + s_l} = \frac{e^{r_w}}{e^{r_w} + e^{r_l}}.$$

Divide top and bottom by $e^{r_w}$:

$$P(y_w \succ y_l) = \frac{1}{1 + e^{-(r_w - r_l)}} = \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big).$$

There it is — the difference of rewards is the log-odds of preference, squashed through a logistic sigmoid. That's clean, and it falls right out of "preference depends on a difference of latent scores." Two things to notice. The probability depends only on the *difference* $r_w - r_l$, so the absolute level of $r$ is unidentifiable — I can add any constant to all rewards and nothing changes. I'll have to pin down that gauge later. And this is literally logistic regression on the reward difference, which means I can fit it by maximum likelihood the usual way. The likelihood of the observed comparison is $\sigma(r_w - r_l)$, so the negative log-likelihood I minimize is

$$\mathrm{loss}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)\sim D}\Big[\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\Big].$$

And let me sanity-check the gradient direction, because a sign error here poisons everything downstream. $\frac{d}{du}\big[-\log\sigma(u)\big] = -(1-\sigma(u)) = \sigma(u) - 1$, with $u = r_w - r_l$. So $\partial\,\mathrm{loss}/\partial r_w = \sigma(u) - 1 < 0$, meaning gradient descent *raises* $r_w$; and $\partial\,\mathrm{loss}/\partial r_l = 1 - \sigma(u) > 0$, so it *lowers* $r_l$. Exactly what I want: push the winner up, the loser down. And the magnitude is $1 - \sigma(u)$ — large when the model currently has it wrong ($u$ small or negative), vanishing once it already ranks the pair correctly with confidence. The loss self-throttles. Good.

What should $r_\theta$ actually *be*, architecturally? I don't want to learn "how to read English" from scratch just to judge responses. So take a model that already understands language — start from the supervised-fine-tuned model — and replace its final unembedding layer (the projection to vocabulary logits) with a small head that outputs a single scalar. The whole transformer does the heavy lifting of comprehension; I'm only learning the readout from "understood text" to "how good is it." A note on size: I might be tempted to make the reward model as big as the policy, but the biggest reward models are costly, unstable to train, and would later have to serve as the value function during RL, where instability is poison. A fixed mid-sized reward model is the safer engineering point: stable enough to initialize the value function and cheap enough to use for every policy size.

Now a practical problem in collecting the comparisons. Pairwise labeling is slow if I literally show two at a time. Better: show a labeler $K$ responses at once (say $K$ between 4 and 9) and have them *rank* all of them. One ranking of $K$ items gives me $\binom{K}{2}$ pairwise comparisons for the price of a single labeling session — a huge multiplier on human throughput. But now how do I feed those into training? The naive thing is to flatten every $\binom{K}{2}$ pair into the dataset as an independent example and shuffle. And that backfires. Each of the $K$ completions then appears in $K-1$ different pairs, so within one epoch the same completion drives $K-1$ separate gradient updates, and worse, all $\binom{K}{2}$ comparisons from one prompt are tightly correlated — they're rankings of the *same* set of responses. The reward model overfits after a single pass; it memorizes the prompt's particular completions rather than learning preference.

The fix is to stop pretending these are independent. Treat all $\binom{K}{2}$ comparisons from a single prompt as *one* batch element. Run each of the $K$ completions through the reward model exactly once to get its scalar — that's $K$ forward passes — and then form all $\binom{K}{2}$ pairwise terms from those cached scalars. Two wins at once. Compute: $K$ forwards instead of the $2\binom{K}{2}$ forwards the naive flattening would do. And generalization: because a prompt's correlated comparisons are now consumed together in one coherent update instead of being smeared across the epoch as fake-independent points, the overfitting pressure from repeated completions is much lower. For one ranked prompt, with pair set $\mathcal P(x)$, the loss averages the pairwise terms:

$$\ell_x(\theta) = -\frac{1}{\binom{K}{2}}\sum_{(y_w,y_l)\in \mathcal P(x)}\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big),\qquad \mathrm{loss}(\theta)=\mathbb E_{x\sim D_{\mathrm{rank}}}[\ell_x(\theta)].$$

Train this for a single epoch — more than one and it overfits fast. And remember the gauge ambiguity: the loss is invariant to shifting all rewards by a constant, so before I do anything with $r$, I'll normalize it with a bias term so that the human demonstration responses get a mean reward of zero. That fixes the arbitrary offset so the reward scale is meaningful going into RL.

Now the payoff stage: I have a cheap, fast $r_\theta$, so let me optimize the policy against it. The objective is simply to make the policy produce high-reward responses: maximize $\mathbb{E}_{x}\,\mathbb{E}_{y\sim\pi_\phi(\cdot\mid x)}\big[r_\theta(x, y)\big]$. Structurally this is the simplest possible RL — a bandit. The "state" is the prompt $x$, the "action" is the entire response $y$, the episode is one step: sample the response, the reward model scores it, episode over. No environment dynamics, no long horizon. (Within the response I'll still treat the token sequence as the thing over which to assign credit, with a value baseline, but there's no external transition structure.)

How do I optimize it? Policy gradient. The score-function estimator says $\nabla_\phi \mathbb{E}[r] = \mathbb{E}\big[r(x,y)\,\nabla_\phi \log \pi_\phi(y\mid x)\big]$ — sample responses, weight each by its reward, push up the log-probability of the high-reward ones. Could I just run vanilla REINFORCE on this? No.

The first reason is the usual RL one: that estimator is high-variance, and an unconstrained gradient step can move the policy a long way in one update. With language models that's catastrophic — the policy can lurch into producing degenerate, repetitive, or off-the-rails text and never recover. I need each update to stay in a trust region: improve the objective, but don't move the policy distribution too far per step. I could do something heavy and constrained, but there's a much cheaper trick. Form the importance ratio $\rho_t = \pi_\phi(a_t\mid s_t)/\pi_{\phi_\text{old}}(a_t\mid s_t)$ between the updated and the data-collecting policy, and instead of maximizing $\rho_t \hat A_t$ — which would happily push $\rho_t$ to infinity for a positive advantage — maximize the *clipped* surrogate

$$\min\!\Big(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\,\hat A_t\Big).$$

Let me check the cases, because the sign matters. If $\hat A_t>0$, increasing $\rho_t$ helps until $\rho_t=1+\epsilon$; beyond that, the clipped term $(1+\epsilon)\hat A_t$ is smaller than $\rho_t\hat A_t$, so the $\min$ chooses the cap and removes the incentive to push higher. If $\hat A_t>0$ and $\rho_t$ is below the interval, the unclipped term is smaller, so the gradient still pushes the probability back up. If $\hat A_t<0$, decreasing $\rho_t$ helps until $\rho_t=1-\epsilon$; below that, multiplication by a negative advantage flips the ordering and the $\min$ chooses $(1-\epsilon)\hat A_t$, so there is no incentive to suppress the token further. If $\hat A_t<0$ and $\rho_t$ is too high, the unclipped term is even worse, so the objective still penalizes the overshoot. The $\min$ is pessimistic in the maximization form: it clips only the changes that would make the surrogate look better than I trust. In code, since I minimize a loss, this same rule appears as $\max(-\rho_t\hat A_t,\ -\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)$. This is a cheap, first-order trust region. I'll use a clip of $\epsilon = 0.2$, run the policy gradient with variance-reduced advantages from a learned value baseline (advantage $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — generalized advantage estimation, the standard variance/bias knob), reusing each batch for only a single inner epoch of minibatch updates. And I'll initialize the value function from the reward model — it already maps text to a scalar, which is most of what predicting return requires, so it's a far better start than random.

The second reason is sharper and specific to optimizing a *learned* reward. My $r_\theta$ is not the true human preference; it's an approximation, and it's only trustworthy where it was trained — on responses near the distribution of the supervised policy that generated the comparison data. If I let the policy roam anywhere in the vast space of token sequences to maximize $r_\theta$, it *will* find regions where $r_\theta$ reports a high score but a real human would recoil — adversarial inputs to the reward model. The optimizer is a relentless adversary against any imperfect reward; it exploits the gaps. So the more I optimize, the more the policy drifts off-distribution and the more the reward model's number decouples from actual preference: the score keeps climbing while real quality stalls or falls, and nothing in the objective tells the optimizer that the high score out there is a lie. This is reward over-optimization, and it's the central danger.

Both reasons point at the same remedy: keep the policy from wandering far from the supervised model, which is exactly the region where the reward model is valid *and* where the policy is well-behaved. So add a penalty for drifting away from the supervised policy $\pi^\text{SFT}$ — specifically a KL divergence $\mathrm{KL}(\pi_\phi \,\|\, \pi^\text{SFT})$. I don't have the full distributions to integrate, but I'm sampling $y \sim \pi_\phi$ anyway, and for a sampled sequence the single-sample estimate of the KL integrand is just the log-ratio $\log \pi_\phi(y\mid x) - \log \pi^\text{SFT}(y\mid x)$ — and over tokens this is a Monte-Carlo estimate of the KL. So fold it straight into the reward, per token. The reward the RL optimizer actually sees becomes

$$R(x, y) = r_\theta(x, y) - \beta\Big(\log \pi_\phi(y\mid x) - \log \pi^\text{SFT}(y\mid x)\Big),$$

with the reward-model score applied at the end of the sequence and the KL term applied as a per-token penalty along the way. This does double duty. It leashes the policy to the region where $r_\theta$ is honest, killing reward hacking, and because it penalizes collapsing onto any single high-reward response (that would blow up the KL), it acts as an entropy-like regularizer that also tames the variance/collapse problem. The coefficient $\beta$ is the leash length. Too small and the policy slips the leash and games the reward model; too large and it's pinned to the supervised model and can't improve. There's a sweet spot — empirically quite small, around $\beta = 0.02$; cranking it up a hundredfold cripples the policy, and zero invites hacking. Putting it together, the objective I maximize is

$$\mathbb{E}_{(x,y)\sim \pi_\phi}\Big[r_\theta(x, y) - \beta\log\frac{\pi_\phi(y\mid x)}{\pi^\text{SFT}(y\mid x)}\Big].$$

So the pipeline has assembled itself out of necessity: supervised fine-tune to get into instruction-following mode and to provide the anchor distribution; a reward model fit with the Bradley-Terry pairwise loss to turn cheap human comparisons into a dense scalar signal; PPO with a KL-to-SFT penalty to optimize that signal without being fooled by it. Each stage exists to break the previous stage's ceiling — imitation can't discriminate, so the reward model adds discrimination; humans can't be queried online, so the reward model is queried instead; the reward model can be gamed, so the KL leash keeps the optimization honest.

Now I run it, and a problem surfaces that I should have anticipated. After RL fine-tuning, I check the model on standard public NLP benchmarks — reading comprehension, QA, translation — and it has *regressed* relative to where the pretrained model was. I've made it more helpful on the prompt distribution at the cost of raw capability elsewhere. That's an alignment tax, and it's bad on its own terms (I don't want a model that obeys but forgets how to do things) and bad strategically: if aligning a model makes it dumber, there's an incentive to deploy the unaligned, more-capable one instead. I want the tax near zero.

My first instinct is that this is just the policy drifting, so tighten the leash — crank $\beta$ up to keep it closer to the supervised model. So I test it: sweep $\beta$ up by a hundredfold. It does *not* fix the regressions, and it tanks the reward (the policy can no longer improve). That's informative. The KL-to-SFT anchor is the wrong anchor for *this* problem. The supervised model has itself already drifted from the broad pretrained model, and pulling toward it doesn't restore the specific capabilities that lived in the pretraining distribution. The capabilities I'm losing are in the *pretraining* data, and my regularizer doesn't point there.

So point there directly. Alongside the RL objective, keep training on the original pretraining objective — maximize the log-likelihood of real pretraining text — mixed into the updates. That anchors the model to the exact distribution where the general capabilities live, instead of to the supervised model that's a step removed from it. The combined objective becomes

$$\mathrm{objective}(\phi) = \mathbb{E}_{(x,y)\sim \pi_\phi}\Big[r_\theta(x, y) - \beta\log\frac{\pi_\phi(y\mid x)}{\pi^\text{SFT}(y\mid x)}\Big] + \gamma\,\mathbb{E}_{x\sim D_\text{pretrain}}\big[\log \pi_\phi(x)\big].$$

In practice I compute the PPO loss and the pretraining negative-log-likelihood loss in consecutive steps and accumulate both gradients before stepping; minimizing $L_{\mathrm{PPO}}+\gamma L_{\mathrm{ptx}}$ is the same sign as maximizing the objective above because $L_{\mathrm{ptx}}=-\mathbb E[\log \pi_\phi(x)]$. The coefficient $\gamma$ trades capability-preservation against alignment reward: I need it reasonably large — around $\gamma = 27.8$, with on the order of $\gamma \gtrsim 20$ enough to counter the worst regressions on the smaller model. I also need enough pretraining data per update for the gradient estimate to be clean: too little (a ratio of about four pretraining examples per RL episode) and the pretraining loss can creep up during training; a lot more (about thirty-two to one) makes the update much slower; eight-to-one is the compromise I'll take. Setting $\gamma = 0$ recovers the plain version without the pretraining mix. The version I want is the one whose gradient contains both forces: preference optimization from PPO and capability retention from the pretraining likelihood.

Let me write the code. The signs have to line up with the optimizer: comparison training minimizes $-\log\sigma(r_w-r_l)$, the token reward uses $-\beta(\log p-\log p_{\mathrm{ref}})$, and PPO minimizes the negative clipped surrogate with a `max`.

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
    # Supervised demonstration fitting and the ptx term are the same next-token NLL;
    # the mask selects response tokens for demonstrations or pretraining tokens for ptx.
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
    # The language backbone reads the prompt-response text; the scalar head emits r(x, y).
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
    # TRL RewardTrainer uses -logsigmoid(chosen - rejected), optionally with a margin.
    # When scores are [prompts, pairs], average pairs within each prompt first.
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
    # TRL's default KL penalty is logp - ref_logp, so reward = -beta * (logp - ref_logp).
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
    # Same GAE recursion as TRL: delta_t = r_t + gamma V_{t+1} - V_t.
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

The whole thing as one causal chain: the pretraining objective is misaligned with user intent and scale doesn't close the gap, so I fine-tune on demonstrations to enter instruction-following mode — but imitation can't tell better from worse and can't exceed the demonstrator. To get a "better-than" signal I'd optimize human preference directly, but humans can't ride inside an RL loop, so I fit a reward model to offline human *comparisons*; comparisons being more reliable than ratings, and the Bradley-Terry model turning "A beats B" into $\sigma(r_w - r_l)$ gives the $-\log\sigma$ pairwise loss. I then optimize the policy against that reward with PPO's clipped surrogate for a cheap trust region — but a learned reward gets gamed under hard optimization, so I add a per-token KL-to-SFT penalty that keeps the policy where the reward model is trustworthy. Finally, RL fine-tuning extracts an alignment tax that tightening KL can't repay, so I mix the pretraining gradient back in to anchor the capabilities while the PPO term keeps pushing on preference.
