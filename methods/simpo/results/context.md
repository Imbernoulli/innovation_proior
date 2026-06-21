# Context: offline preference optimization for language models (circa 2023-2024)

## Research question

We have a pretrained, supervised-fine-tuned language model (an SFT policy `pi_theta(y|x)`) and a
static dataset of human preference pairs `(x, y_w, y_l)` — a prompt, a winning (preferred) response,
and a losing (dispreferred) one. The goal is to nudge the policy so that, at generation time, it
produces responses more like the winning ones and less like the losing ones, *without* running a
full reinforcement-learning loop.

How can offline preference optimization be made more memory- and compute-efficient while keeping
the training objective well-aligned with how the model actually ranks responses at generation time?

## Background

By this time, aligning LLMs to human preferences is the dominant post-training step. The field state
rests on a few load-bearing pieces.

**The Bradley-Terry preference model (Bradley & Terry 1952).** Human pairwise preferences are modeled
as arising from a latent scalar reward `r*(x,y)`: the probability that `y_1` beats `y_2` is
`p*(y_1 > y_2 | x) = sigma(r*(x,y_1) - r*(x,y_2))`, with `sigma` the logistic function. Fitting a
reward model `r_phi` to a preference dataset is then just maximum-likelihood binary classification,
`L_R = -E[log sigma(r_phi(x,y_w) - r_phi(x,y_l))]`. Only *differences* of reward enter, so a reward
function is identifiable only up to an additive function of `x`.

**KL-constrained reward maximization.** The RL step of the classical pipeline maximizes expected
reward while staying close to the reference policy:
`max_pi E_{x, y~pi}[r(x,y)] - beta * KL(pi(.|x) || pi_ref(.|x))`. The KL leash is what keeps the
policy on the distribution where the reward is trustworthy and prevents mode collapse onto a few
high-reward strings. This constrained problem has a known closed-form optimum (Gibbs / variational):
`pi_r(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(x,y)/beta)`, where `Z(x)` is an intractable partition
function summing over all sequences.

**The generation metric.** When the policy actually produces text, candidates are scored by the
*average* per-token log-probability, `p_theta(y|x) = (1/|y|) sum_i log pi_theta(y_i | x, y_<i)` — this
is the quantity beam search and multiple-choice-style scoring rank by. It involves the policy alone;
the reference model never appears at inference.

**Margins and generalization.** A long-standing observation in classification is that enforcing a
*margin* between classes — not just getting the sign of the decision right, but separating the classes
by a gap — improves generalization (the max-margin principle behind support-vector machines; the
"home advantage" term in Bradley-Terry sports models). In the preference setting the two "classes" are
the winning and losing responses for one prompt.

A few diagnostic phenomena about *existing* preference-trained models are also part of the landscape:

- **Length bias.** Preference-optimized LLMs tend to produce verbose outputs, and longer responses
  are not reliably better; verbosity is a known way preference objectives get exploited. Mechanically,
  the *summed* log-probability of a response decreases with length (more tokens, each contributing a
  negative log-prob), so any objective that rewards higher summed log-prob on the winning response can
  be satisfied by inflating probabilities on long sequences when the winner happens to be longer.
- **Training-vs-generation ranking in reference-based offline training.** For a reference-based
  offline objective, satisfying its training-time reward ordering on a triple does not guarantee that
  the policy assigns higher average log-likelihood to the winner than the loser; in practice only
  about half of the training triples end up satisfying the average-log-likelihood ordering after such
  training, and concurrent analyses report near-chance ranking accuracy by that metric even after
  extensive optimization.
- **The math-reasoning failure mode (Pal et al. 2024).** On preference pairs where flipping a single
  token flips the label — `2+2=4` (chosen) vs `2+2=5` (rejected), which share almost all tokens — a
  contrastive objective can *increase the reward margin while decreasing the absolute likelihood of
  the chosen sequence*, because pushing the rejected sequence's probability down drags the nearly
  identical chosen sequence down with it. Adding back a reference-calibrated supervised term mitigates
  this. This is the regime that matters when preferences are over math solutions.

## Baselines

These are the prior offline objectives a new method would be measured against and react to. Each maps
preference data directly onto a policy loss; they differ in what reward they reparameterize and what
they must keep in memory.

**DPO — Direct Preference Optimization (Rafailov et al. 2023).** Start from the KL-constrained optimum
`pi_r(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(x,y)/beta)`. Taking logs and rearranging expresses the reward
in terms of the policy: `r(x,y) = beta log[pi_r(y|x)/pi_ref(y|x)] + beta log Z(x)`. Because the
Bradley-Terry model depends only on reward *differences*, substituting this reparameterized reward
into `p*(y_w > y_l | x) = sigma(r(x,y_w) - r(x,y_l))` makes the intractable `beta log Z(x)` cancel,
yielding a loss purely in policy log-ratios:

```
L_DPO = - E_{(x,y_w,y_l)~D} [ log sigma( beta log[pi_theta(y_w|x)/pi_ref(y_w|x)]
                                        - beta log[pi_theta(y_l|x)/pi_ref(y_l|x)] ) ].
```

This bypasses both the explicit reward model and the RL loop — a single maximum-likelihood objective.
Its gradient up-weights triples the implicit reward currently mis-ranks and moves
`nabla log pi(y_w) - nabla log pi(y_l)`. The reference policy `pi_ref` must be held in memory and
run on every batch to compute the log-ratios. The implicit reward is a log-ratio to `pi_ref`;
the gradient term `nabla log pi(y_w) - nabla log pi(y_l)` uses the *summed* log-prob gradient,
not normalized by length.

**IPO — Identity Preference Optimization (Azar et al. 2023).** Replaces the Bradley-Terry sigmoid
(which can saturate and overfit when preferences are near-deterministic) with a squared-loss objective
that regresses the policy's log-ratio gap toward a fixed target tied to `1/(2 tau)`. It builds in a
*target separation* between winner and loser rather than pushing the gap to infinity, while keeping
the reference model `pi_ref` in the log-ratios.

**ORPO — Odds-Ratio Preference Optimization (Hong et al. 2024).** A reference-free objective: it adds,
to the ordinary supervised fine-tuning loss on the winner, a penalty on the log odds-ratio between
winning and losing responses, `log[odds(y_w)/odds(y_l)]` with `odds(y) = pi(y)/(1-pi(y))`. No
reference model is needed.

**Reward-model + PPO (the classical pipeline, Ouyang et al. 2022; Schulman et al. 2017).** Fit `r_phi`
by Bradley-Terry MLE, then maximize `E[r_phi] - beta KL` on-policy with PPO. Three models are in
play (policy, reference, reward), with an actor-critic loop and substantial engineering overhead.

## Evaluation settings

The natural yardsticks already in use for offline preference optimization:

- **Chat / instruction-following win rates.** AlpacaEval 2 (length-controlled and raw win rate against
  a strong reference model, judged by an LLM) and the harder Arena-Hard benchmark; MT-Bench. Response
  length is reported alongside, since length exploitation is a known confound (the length-controlled
  win rate exists precisely to discount verbosity).
- **Math-reasoning accuracy, judge-free.** Grade-school and competition math benchmarks — GSM8K
  (~1.3K problems), the MATH-500 subset, and AIME 2024 (30 problems) — scored by a symbolic answer
  checker (sympy / math-equality), with greedy decoding (temperature 0). No LLM judge is involved, so
  the metric is exact-match accuracy of the final answer.
- **General-capability retention.** The HuggingFace Open LLM Leaderboard tasks (MMLU, ARC, HellaSwag,
  TruthfulQA, Winograd/Winogrande, GSM8K), to check that preference optimization does not erode
  knowledge or reasoning.
- **Training setup.** A math-specialized SFT model as the base (e.g. a 1.5B math-instruct model with a
  fixed chat template); a preference dataset of math problems with chosen/rejected full solutions used
  as response-level pairs; full-parameter fine-tuning on a few GPUs with ZeRO/DeepSpeed sharding;
  cosine schedule with warmup; a small learning rate; a handful of epochs. The per-method scalars
  (the reward-scaling `beta`, the margin, the learning rate) are tuned per method, since preference
  objectives are known to be sensitive to them.

## Code framework

The objective plugs into a standard paired-preference training harness. The data pipeline yields, per
example, a prompt with a chosen and a rejected continuation; a forward pass through the policy gives
the per-token log-probabilities of each response; a reducer turns those into a per-sequence summary;
and a trainer turns the chosen/rejected summaries into a scalar loss and backpropagates. What summary
to compute from the log-probs, and what loss to compute from the summaries, is exactly what is to be
designed — so the substrate is only the generic machinery that already exists, with one empty slot for
the preference loss.

```python
import torch
import torch.nn.functional as F


def per_token_logps(logits, labels, ignore_index=-100):
    """Standard reducer: gather the log-prob of each gold token under the policy.
    Returns the summed log-prob over response tokens and the response length."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    mask = labels != ignore_index
    labels[~mask] = 0
    token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    summed = (token_logps * mask).sum(-1)          # sum_i log pi(y_i | x, y_<i)
    length = mask.sum(-1)                           # |y| = number of response tokens
    return summed, length


def sequence_score(summed_logp, length):
    # TODO: the per-sequence summary we will compute from a response's token log-probs.
    #       Decide what scalar of (summed_logp, length) the preference signal is built on.
    pass


def compute_preference_loss(policy_chosen_logps, policy_rejected_logps, beta, **kwargs):
    # TODO: the preference loss we will design.
    #       Given the per-sequence summaries of the chosen and rejected responses
    #       (and the scaling beta, plus any extra scalars the objective needs),
    #       return the per-example loss to minimize.
    pass


# existing paired-preference training loop the objective plugs into
def train_step(model, batch, optimizer, beta, **kwargs):
    optimizer.zero_grad()
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    summed, length = per_token_logps(logits, batch["labels"])
    n = summed.shape[0] // 2
    chosen = sequence_score(summed[:n], length[:n])
    rejected = sequence_score(summed[n:], length[n:])
    loss = compute_preference_loss(chosen, rejected, beta, **kwargs).mean()
    loss.backward()
    optimizer.step()
    return loss
```

The harness supplies one (summed log-prob, length) pair per response; `sequence_score` and
`compute_preference_loss` are the two slots the objective will fill.
