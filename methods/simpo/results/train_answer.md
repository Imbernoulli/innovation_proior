We have a supervised-fine-tuned language model $\pi_\theta(y|x)$ and a static pile of human preference pairs $(x, y_w, y_l)$ — a prompt, a winning response, a losing one — and we want to nudge the policy toward producing more of the winners and fewer of the losers, without standing up a reinforcement-learning loop. The offline-without-RL route already exists: reparameterize the KL-constrained reward optimum, cancel the intractable partition function, and arrive at a clean maximum-likelihood loss in policy log-ratios. That is DPO, and it is the right baseline to stare at, because two of its pieces bother me. The first is cost. Its reward is $\beta \log[\pi_\theta(y|x)/\pi_\mathrm{ref}(y|x)]$, so the reference policy $\pi_\mathrm{ref}$ sits inside every term; at train time I must keep a frozen copy of the model resident and run a second forward pass on every batch just to compute a baseline I subtract off — a whole extra model's footprint and a doubled forward cost on a memory-tight, multi-billion-parameter box. The second bothers me more. At generation there is no reference model anywhere: I decode from $\pi_\theta$ and rank candidates — beam search, multiple-choice scoring — by the policy's *average* per-token log-likelihood $p_\theta(y|x) = \frac{1}{|y|}\sum_i \log \pi_\theta(y_i \mid x, y_{<i})$. That average is the thing the model is graded by when it produces text, yet DPO optimizes a log-*ratio* against the reference, a different function of the same response. The mismatch can bite concretely: if a triple satisfies DPO's reward ordering $r(x,y_w) > r(x,y_l)$, expanding and rearranging gives $\log \pi_\theta(y_w) - \log \pi_\theta(y_l) > \log \pi_\mathrm{ref}(y_w) - \log \pi_\mathrm{ref}(y_l)$ — a condition on raw summed log-probs offset by whatever the reference assigns, with no reason it implies the generation condition $\frac{1}{|y_w|}\log\pi_\theta(y_w) > \frac{1}{|y_l|}\log\pi_\theta(y_l)$. The reference offset is arbitrary and the lengths differ, so the two orderings disagree on many triples; DPO diagnostics indeed show its reward ordering looking correct while average-log-likelihood ranking hovers near chance. IPO keeps the same reference and only swaps the logistic loss for a squared regression; ORPO drops the reference but contrasts an odds-ratio plus an SFT term rather than the per-token metric the model is actually judged by. None of them optimizes the quantity used at decode time, and most still pay for $\pi_\mathrm{ref}$.

The clean thing to want, then, is to make the reward I optimize *be* the metric the model is generated and ranked by — not optimize one quantity and hope it transfers. I propose SimPO, Simple Preference Optimization. Its first component is a reference-free, length-normalized reward equal to the policy's own average per-token log-likelihood, which is exactly the generation-ranking metric:
$$r_\mathrm{SimPO}(x,y) = \frac{\beta}{|y|}\log\pi_\theta(y|x) = \frac{\beta}{|y|}\sum_i \log\pi_\theta(y_i \mid x, y_{<i}).$$
The naive reference-free reward would be the *summed* log-prob $\beta\log\pi_\theta(y|x)$, and it fails for a structural reason: every extra token contributes another $\log\pi_\theta(y_i)\le 0$, so longer sequences carry a systematically lower summed log-prob, and when $y_w$ happens to be longer than $y_l$ the only way to push its summed log-prob above the loser's is to inflate token probabilities on the long winning sequence — baking "long = good" into the model as an artifact of the reward's length-sensitivity. The $1/|y|$ is precisely the fix: it cancels that handicap by scoring every response per token, putting long and short on the same footing, and it makes the reward *identical in kind* to the inference metric. So the train/generation mismatch and the length bias have one shared cure, and averaging is it. Dropping $\pi_\mathrm{ref}$ costs the explicit KL leash, but the training regime supplies a practical one — start from a good SFT model, small learning rate, few epochs, diverse preference data — enough reason to try the policy-only reward before paying for a frozen reference on every batch.

The second component comes from sitting with the Bradley-Terry loss that results from plugging this reward in, $-\mathbb{E}[\log\sigma(r(x,y_w) - r(x,y_l))]$. It only asks for $r(y_w) > r(y_l)$ — it is satisfied the instant the winner's reward exceeds the loser's by an infinitesimal amount. But getting the sign right is not the same as generalizing well; the max-margin lesson from classification (and the additive "home-advantage" offset in Bradley-Terry sports models) says a comfortable separation between classes generalizes better than a barely-correct one. The two classes here are the winning and losing responses for one prompt, so I demand the gap exceed a target $\gamma > 0$ before the model is satisfied, by putting the margin into the preference model itself:
$$p(y_w > y_l \mid x) = \sigma\!\left(r(x,y_w) - r(x,y_l) - \gamma\right).$$
The $-\gamma$ shifts the sigmoid so the loss is not near-minimal until $r(y_w) - r(y_l)$ has crossed $\gamma$, keeping the winner pulled above the loser until there is a real cushion. Substituting $r_\mathrm{SimPO}$ gives the final objective
$$\mathcal{L}_\mathrm{SimPO} = -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right].$$
Checking the gradient confirms both choices pull their weight. With $u_\gamma$ the argument of the sigmoid, $-\,d\log\sigma(u_\gamma)/du_\gamma = -\sigma(-u_\gamma)$ and $\nabla_\theta u_\gamma = \beta\big(\frac{1}{|y_w|}\nabla\log\pi_\theta(y_w) - \frac{1}{|y_l|}\nabla\log\pi_\theta(y_l)\big)$, so
$$\nabla\mathcal{L}_\mathrm{SimPO} = -\,\beta\,\mathbb{E}\!\left[s_\theta\left(\frac{1}{|y_w|}\nabla\log\pi_\theta(y_w|x) - \frac{1}{|y_l|}\nabla\log\pi_\theta(y_l|x)\right)\right],\quad s_\theta = \sigma\!\left(\frac{\beta}{|y_l|}\log\pi_\theta(y_l) - \frac{\beta}{|y_w|}\log\pi_\theta(y_w) + \gamma\right).$$
The per-example weight $s_\theta$ is large exactly when the policy wrongly assigns higher average log-likelihood to the loser (plus the margin), which is the right thing to up-weight, and it contains no reference model — unlike DPO's weight, which needs $\pi_\mathrm{ref}$. Each response's gradient is divided by its own length, so a response with twice the tokens no longer gets twice the update and cannot dominate a batch, where DPO's un-normalized $\nabla\log\pi(y_w) - \nabla\log\pi(y_l)$ reopens the length channel. The two design choices are orthogonal: length normalization fixes *what* the reward is (per-token, generation-aligned, reference-free), the margin fixes *how hard* to separate the classes. Dropping the normalization but keeping the margin reverts to summed-log-prob reward maximization and reintroduces length bias; dropping the margin but keeping normalization falls back to "any positive gap will do." The margin $\gamma$ is a tuned knob — too small and I am back to merely asking for the right sign, too large and I demand an unrealistic per-token gap and over-suppress fluent losing responses; chat settings use roughly $\beta \in [2, 2.5]$ with $\gamma$ swept over a small grid. One regime deserves a flag: when winning and losing solutions are nearly identical — $2{+}2{=}4$ versus $2{+}2{=}5$, one token apart — a contrastive objective can widen the reward margin while the *absolute* likelihood of the chosen sequence falls, because pushing the near-identical rejected sequence down drags the chosen down too, and the margin term does not rescue this on its own. The clean remedy is an optional supervised anchor on the winner, $-\lambda\log\pi_\theta(y_w|x)$ inside the loss, rewarding the model for keeping the winner's probability up in absolute terms; I leave the core objective as the two-component reference-free margin loss and keep the SFT anchor as an add-on for the reasoning-heavy regime.

The harness supplies, per response, the summed token log-prob and the length. The first slot, `sequence_score`, is where the reward's shape lives — dividing summed log-prob by length is the length normalization and the alignment with generation in one step; the second slot, `compute_preference_loss`, forms the chosen/rejected difference, subtracts the margin, scales by $\beta$, and runs it through $-\log\sigma$, carrying the margin as $\gamma/\beta$ so the single $\beta$-multiply reproduces $\beta(\Delta) - \gamma$ exactly.

```python
import torch
import torch.nn.functional as F


def per_token_logps(logits, labels, ignore_index=-100):
    """Return (summed log-prob over response tokens, response length |y|)."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    mask = labels != ignore_index
    labels[~mask] = 0
    token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    summed = (token_logps * mask).sum(-1)      # sum_i log pi(y_i | x, y_<i)
    length = mask.sum(-1)                       # |y|
    return summed, length


def sequence_score(summed_logp, length):
    """Length normalization: average log-likelihood = generation metric, reference-free."""
    return summed_logp / length                # (1/|y|) log pi(y|x)


def simpo_loss(policy_chosen_logps, policy_rejected_logps, beta=2.0, gamma=1.0):
    """SimPO loss from per-sequence AVERAGE log-probs of chosen/rejected responses.

    L = -log sigma( beta*(avg_w - avg_l) - gamma )
      = -log sigma( beta*((avg_w - avg_l) - gamma/beta) )
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps   # avg_w - avg_l
    gamma_logratios = gamma / beta                               # full margin -> code-space ratio
    logits = pi_logratios - gamma_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = (beta * policy_chosen_logps).detach()       # length-normalized implicit reward
    rejected_rewards = (beta * policy_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def train_step(model, batch, optimizer, beta=2.0, gamma=1.0):
    optimizer.zero_grad()
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    summed, length = per_token_logps(logits, batch["labels"])
    n = summed.shape[0] // 2
    chosen_avg = sequence_score(summed[:n], length[:n])
    rejected_avg = sequence_score(summed[n:], length[n:])
    losses, _, _ = simpo_loss(chosen_avg, rejected_avg, beta=beta, gamma=gamma)
    loss = losses.mean()
    loss.backward()
    optimizer.step()
    return loss
```
