Let me start from what's actually bothering me: cost and a nagging mismatch. I have an SFT'd language model and a pile of preference pairs `(x, y_w, y_l)`, and I want to teach the model to lean toward the winners. The offline-without-RL route already exists and works: reparameterize the KL-constrained-reward optimum, cancel the partition function, and get a clean maximum-likelihood loss in policy log-ratios. That's DPO, and I'll write it down because it's my baseline and because I want to stare at exactly which pieces of it are paying their way.

The derivation, so I have it cold. The RL objective is `max_pi E_{y~pi}[r(x,y)] - beta KL(pi || pi_ref)`. Rewrite the KL inside the expectation: this is `min_pi E_{y~pi}[ log(pi/pi_ref) - r/beta ]`. Now complete the square by defining `pi*(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(x,y)/beta)` with `Z(x) = sum_y pi_ref(y|x) exp(r/beta)` — then `log(pi/pi_ref) - r/beta = log(pi/pi*) - log Z(x)`, and since `log Z` doesn't depend on `pi`, the objective is `min_pi E_x[ KL(pi || pi*) - log Z(x) ]`. KL is minimized at zero when `pi = pi*`, by Gibbs. So the optimal policy is `pi_r = (1/Z) pi_ref exp(r/beta)`. Invert it: `r(x,y) = beta log[pi_r(y|x)/pi_ref(y|x)] + beta log Z(x)`. Plug that reparameterized reward into Bradley-Terry, `p(y_w > y_l) = sigma(r(x,y_w) - r(x,y_l))`. The reward difference kills the `beta log Z(x)` term — it's the same `x` for both responses — and out drops

`L_DPO = - E[ log sigma( beta log[pi(y_w)/pi_ref(y_w)] - beta log[pi(y_l)/pi_ref(y_l)] ) ]`.

That is the clean part, and it explains why everyone uses it. Now the nagging. Two things.

First, `pi_ref`. It sits in every term. At train time I have to keep a frozen copy of the model resident and run a second forward pass on every batch to get those reference log-probs. On a multi-billion-parameter model on a memory-tight box, that's a whole extra model's worth of footprint and a doubled forward cost, purely to compute a baseline I subtract off. Is it earning its keep? It comes from the KL leash — `pi_ref` is what the policy is regularized toward. Maybe I need it, maybe I don't. Park that.

Second, and this is the one that actually itches. What does the reward I'm optimizing have to do with how the model gets used? At generation, there is no reference model anywhere. I decode from `pi_theta` and I rank candidates — beam search, multiple-choice scoring — by the policy's *average* per-token log-likelihood, `p_theta(y|x) = (1/|y|) sum_i log pi(y_i | x, y_<i)`. That average log-likelihood *is* the thing the model is graded by when it produces text. But the reward DPO optimizes is `r(x,y) = beta log[pi(y)/pi_ref(y)]`, a log-*ratio* against the reference. Those are different functions of the same response.

Let me make sure that difference can actually bite. Suppose on a triple I've made `r(x,y_w) > r(x,y_l)`, which is what the DPO loss rewards. Expand: `beta log pi(y_w) - beta log pi_ref(y_w) > beta log pi(y_l) - beta log pi_ref(y_l)`. Rearrange to `log pi(y_w) - log pi(y_l) > log pi_ref(y_w) - log pi_ref(y_l)`. So the *training* condition is on raw summed log-probs, *offset by whatever the reference assigns*. The *generation* condition I actually care about is `p_theta(y_w) > p_theta(y_l)`, i.e. `(1/|y_w|) log pi(y_w) > (1/|y_l|) log pi(y_l)` — averaged, no reference. There's no reason satisfying the first implies the second. The reference offset can be anything, and the lengths `|y_w|`, `|y_l|` can differ, so the two orderings can disagree on the same triple. Existing DPO diagnostics already show this failure mode: the reward ordering can look correct while the average-log-likelihood ordering hovers near chance. That is exactly the kind of disconnect that would let training look like it's working while the generation-time ranking barely moves. The reward I optimize and the metric I'm judged by are not the same object.

So here's the cleaner thing to want: make the reward I optimize *be* the metric the model is generated/ranked by. Don't optimize one quantity and hope it transfers to another. The generation metric is the average log-likelihood. So let me try to build the preference reward directly out of that.

First instinct, the naive one: just use the policy's log-probability of the response as the reward, drop the reference entirely. `r(x,y) = beta log pi(y|x) = beta sum_i log pi(y_i | x, y_<i)`. Reference-free for free — no `pi_ref`, the memory and compute problem evaporates. Plug into Bradley-Terry: `L = - E[ log sigma( beta log pi(y_w) - beta log pi(y_l) ) ]`. Clean. But wait — is this the generation metric? It's the *summed* log-prob, not the average. And summed log-prob has a structural problem with length. Every extra token contributes another `log pi(y_i) <= 0`, so longer sequences have systematically *lower* summed log-prob. Now look at what the loss does when `y_w` happens to be longer than `y_l`: to push `log pi(y_w) > log pi(y_l)`, the model has to overcome a length handicap, and the only lever it has is to crank up token probabilities on the long winning sequence — inflate probabilities precisely to compensate for length. That's a recipe for degeneration and for baking a length preference into the model: it learns "long = good" as an artifact of the reward's length-sensitivity, not because long is actually preferred. And it's a known sore spot — preference-tuned models drift verbose, and verbosity isn't quality. So summed log-prob is the wrong reward.

But the fix is sitting right there in what I already said I want. The generation metric isn't the summed log-prob, it's the *average* — `(1/|y|) log pi(y|x)`. The `1/|y|` is exactly the length normalization that cancels the structural length handicap: every response is scored per-token, so a long response and a short response are on the same footing, and there's no incentive to inflate probabilities just to beat a length penalty. And it's the *same* quantity the model is ranked by at decode time. So the two problems — the train/generation mismatch and the length bias — have one fix, and it's the same fix: use the average log-likelihood as the reward.

`r_SimPO(x,y) = (beta/|y|) log pi(y|x) = (beta/|y|) sum_i log pi(y_i | x, y_<i)`.

Reference-free, because it's just the policy's own per-token score. Aligned with generation, because it *is* the generation-ranking metric (scaled by `beta`). And length-debiased, because the `1/|y|` puts winner and loser on a per-token basis. Those are the three constraints I wanted.

Let me double-check the reference question — do I actually need `pi_ref` for anything once I've done this? The reference was there for the KL leash, to keep the policy from running off to degenerate high-reward strings. If I drop it I lose the explicit regularizer. But the training regime can still act as a practical leash: start from a good SFT model, use a small learning rate, make only a few passes, and train on diverse preference data. That is not a theorem, so I should watch drift, but it is enough reason to try the simpler policy-only reward before paying for a frozen reference model in every batch. If this turns out too loose, a supervised anchor can come back as a separate stabilizer.

Now plug `r_SimPO` straight into Bradley-Terry, exactly as DPO did with its reward:

`L = - E[ log sigma( (beta/|y_w|) log pi(y_w|x) - (beta/|y_l|) log pi(y_l|x) ) ]`.

Let me check the gradient does the right thing before I trust it. Write `u = (beta/|y_w|) log pi(y_w) - (beta/|y_l|) log pi(y_l)`, loss `= -log sigma(u)`. Then `d(-log sigma(u))/du = -(1 - sigma(u)) = -sigma(-u)`, and `nabla_theta u = beta( (1/|y_w|) nabla log pi(y_w) - (1/|y_l|) nabla log pi(y_l) )`. So

`nabla L = - beta * sigma(-u) * ( (1/|y_w|) nabla log pi(y_w) - (1/|y_l|) nabla log pi(y_l) )`.

Two things to notice, and both are improvements over DPO's gradient. The per-example weight is `sigma(-u) = sigma( (beta/|y_l|) log pi(y_l) - (beta/|y_w|) log pi(y_w) )` — it's large exactly when the policy *wrongly* assigns higher average log-likelihood to the loser than the winner, which is the right thing to up-weight, and it contains no reference model at all (DPO's weight `sigma(beta log[pi(y_l)/pi_ref(y_l)] - beta log[pi(y_w)/pi_ref(y_w)])` needs `pi_ref`). And the two log-prob gradients are each divided by their own length, so a long response and a short one push with comparable magnitude. DPO's gradient moves `nabla log pi(y_w) - nabla log pi(y_l)` un-normalized, so a response with twice the tokens gets roughly twice the gradient and can dominate a batch — there's the length bias again, now visible in the gradient. Length normalization removes it. Good, the reward choice and the gradient agree.

Is this enough? I have a reference-free, generation-aligned, length-debiased objective. But sit with the Bradley-Terry loss for a second. All it asks is `r(y_w) > r(y_l)` — it's happy the instant the winner's reward exceeds the loser's by an *infinitesimal* amount. The sigmoid keeps pushing, sure, but there's no notion of a *target* gap. Getting the sign right is a weak requirement, and I know from plain classification that getting the sign right is not the same as generalizing well — the lesson of margins. A classifier that separates the training classes by a comfortable gap generalizes better than one that just barely separates them; that's the whole max-margin idea behind SVMs, and in Bradley-Terry ranking models there's even a name for an additive offset that gives one side a built-in edge — the "home advantage." Here the two "classes" are the winning and losing responses for a single prompt. I'd like to insist not merely that the winner outscore the loser, but that it outscore it by at least some target amount.

So put a margin into the preference model itself. Demand that the reward gap exceed `gamma > 0` before the model is "satisfied":

`p(y_w > y_l | x) = sigma( r(x,y_w) - r(x,y_l) - gamma )`.

The `-gamma` shifts the sigmoid: now the loss isn't near-minimal until `r(y_w) - r(y_l)` has crossed `gamma`, not just zero. It keeps pulling the winner above the loser until there's a real cushion. Plug in `r_SimPO`:

`L_SimPO = - E[ log sigma( (beta/|y_w|) log pi(y_w|x) - (beta/|y_l|) log pi(y_l|x) - gamma ) ]`.

The gradient direction is the same length-normalized one I already checked, because `gamma` is constant with respect to `theta`; what changes is the weight. If `u_gamma = (beta/|y_w|) log pi(y_w) - (beta/|y_l|) log pi(y_l) - gamma`, then `-d log sigma(u_gamma)/d u_gamma = -sigma(-u_gamma)`, so the weight becomes `sigma((beta/|y_l|) log pi(y_l) - (beta/|y_w|) log pi(y_w) + gamma)`. The margin therefore keeps hard or under-separated pairs weighted longer, without bringing the reference model back.

How big should `gamma` be? It's a knob with two sides. If it is too small, I am almost back to merely asking for the right sign. If it is too large, I am demanding an unrealistic per-token average-log-likelihood gap and forcing the model to suppress losing responses that may still be fluent and useful. So `gamma` is something to tune, not a constant to derive once. (IPO also bakes in a target gap, but it does it through a squared regression of the reference-based log-ratio toward a fixed value — it keeps the reference model and uses a different loss shape. The margin idea is shared; doing it as an additive bias on a length-normalized reference-free reward inside the logistic loss is what keeps mine simple.)

Let me re-examine the two design choices once more, adversarially. The length normalization: what if I drop it and keep just the margin, `- E[log sigma(beta log pi(y_w) - beta log pi(y_l) - gamma)]`? That's reward maximization on summed log-probs — essentially a contrastive objective without any supervised anchor — and it reopens the same length channel I am trying to close. The normalization is load-bearing, not cosmetic. The margin: what if I drop it and keep just normalization? I lose the separation cushion and fall back to "any positive gap will do," weaker generalization. So both pieces pull their weight, and they're orthogonal — one fixes *what* quantity the reward is (per-token, generation-aligned, reference-free), the other fixes *how hard* to separate the classes.

One more sanity check on consistency with DPO, because I want to know exactly what I changed. DPO's reward is `beta log[pi/pi_ref]`; mine is `(beta/|y|) log pi`. So the skeleton is still Bradley-Terry over a policy-defined reward, but I have removed the reference log-probability, normalized the policy log-probability by response length, and added the fixed margin. Nothing exotic; I removed a model and changed the reward to be the metric I'm scored by.

I should flag one regime where I'd want to watch this carefully, because it's exactly the setting I might be deploying into: preferences over *math* solutions, where the winning and losing responses can be nearly identical — `2+2=4` vs `2+2=5`, differing in one token. A contrastive objective there can do a perverse thing: it widens the *reward margin* by pushing the loser's probability down, but because the chosen sequence shares almost every token with the rejected one, dragging the rejected down drags the chosen down too — the absolute likelihood of the *correct* answer can fall even as the margin grows. My margin term, if anything, asks for a bigger gap, so it will not rescue this on its own. The clean remedy is to anchor the chosen sequence with a supervised term — add `lambda log pi(y_w|x)` inside the maximized score, equivalently `-lambda log pi(y_w|x)` in the minimized loss — so the model is rewarded for keeping the winner's probability up in absolute terms, not just relative to the loser. I'll leave the core objective as the two-component reference-free margin loss, and keep the SFT anchor as an optional add-on for the reasoning-heavy regime where likelihood-of-chosen actually matters.

The harness already gives me, per response, the summed token log-prob and the length. The first slot, `sequence_score`, is where the reward's *shape* lives: I divide summed log-prob by length to get the average — that single division is the length normalization and the alignment-with-generation, all at once. The second slot, `compute_preference_loss`, takes the chosen and rejected averages, forms their difference, subtracts the margin, scales by `beta`, and runs it through `-log sigma`. The implementation nicety is to subtract `gamma/beta` before the final multiply, so `beta*((avg_w - avg_l) - gamma/beta)` reproduces `beta*(avg_w - avg_l) - gamma` exactly. That's it; the contribution is two lines.

```python
import torch
import torch.nn.functional as F


def per_token_logps(logits, labels, ignore_index=-100):
    """Gather log pi(y_i | x, y_<i) for each gold token; return summed log-prob and length."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    mask = labels != ignore_index
    labels[~mask] = 0
    token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    summed = (token_logps * mask).sum(-1)   # sum_i log pi(y_i | x, y_<i)
    length = mask.sum(-1)                    # |y|
    return summed, length


def sequence_score(summed_logp, length):
    # the reward shape: AVERAGE log-likelihood = the generation-ranking metric, length-debiased,
    # reference-free.  r_SimPO(x,y)/beta = (1/|y|) log pi(y|x)
    return summed_logp / length


def compute_preference_loss(chosen_avg_logp, rejected_avg_logp, beta, gamma):
    # Bradley-Terry with a length-normalized reference-free reward and a target margin gamma:
    #   L = -log sigma( beta*(avg_w - avg_l) - gamma )
    # carry the margin as gamma/beta so the single beta-multiply reproduces (beta*delta - gamma).
    logits = (chosen_avg_logp - rejected_avg_logp) - gamma / beta
    return -F.logsigmoid(beta * logits)


def train_step(model, batch, optimizer, beta=2.0, gamma=1.0):
    optimizer.zero_grad()
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    summed, length = per_token_logps(logits, batch["labels"])
    n = summed.shape[0] // 2
    chosen = sequence_score(summed[:n], length[:n])      # average log-prob of chosen
    rejected = sequence_score(summed[n:], length[n:])    # average log-prob of rejected
    loss = compute_preference_loss(chosen, rejected, beta, gamma).mean()
    loss.backward()
    optimizer.step()
    return loss
```

The whole chain: I was paying for a reference model and, worse, optimizing a log-ratio reward that isn't the average-log-likelihood the model is actually ranked by at generation — so the two orderings can disagree on many triples. Make the reward *be* the generation metric: the policy's own per-token average log-probability. That single move drops the reference model, aligns training with inference, and — because averaging divides out the structural length penalty of summed log-probs — closes the length-bias channel, which also shows up as length-normalized, reference-free gradient weights. Then, because Bradley-Terry only asks for the right sign and margins are what give generalization, add a target reward margin `gamma` as an additive bias in the logistic objective and tune it. Two components — average-log-likelihood reward, target margin — and the loss is `-log sigma( (beta/|y_w|) log pi(y_w) - (beta/|y_l|) log pi(y_l) - gamma )`.
