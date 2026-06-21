We have a large teacher that reasons well — it solves competition math — and a small student we can actually afford to serve, and we want to transfer the teacher's behavior into the student at a fraction of the inference cost. The naive recipe, which is also the standard one, is to take a pile of solutions (the teacher's, or even ground-truth) and fine-tune the student to predict them token by token: maximum likelihood. It half-works and then it stalls, and the stall has two distinct causes that demand two distinct fixes. The first is about the *data distribution*. The student is auto-regressive: at inference each token is conditioned on the student's own prior tokens, so the distribution of partial sequences it sees is determined by itself. But during fixed-dataset training every prefix $y_{<n}$ was a clean teacher prefix; the student never practiced continuing from one of *its own* prefixes. Being smaller, it will eventually make an early slip the teacher never makes — a wrong sign, a dropped term — and the instant it does it is standing in a state with zero support in its training data and improvises, and because each token conditions on the last the drift compounds to the end of the solution. This is exactly the cascade that behavior cloning suffers in imitation learning, the train-on-the-expert's-states, test-on-your-own mismatch that DAgger was built to fix; and it afflicts *any* method trained on a fixed set of sequences, teacher-generated or not. The second cause is about the *objective*. Maximum likelihood on soft teacher targets minimizes, per position, the forward KL $D_{KL}(p_T \| p_S) = \sum_v p_T(v)\log\!\big(p_T(v)/p_S(v)\big)$. The weight on each token is the *teacher's* probability $p_T(v)$, and as $p_S(v)\to 0$ with $p_T(v)>0$ the term diverges to $+\infty$: forward KL is mass-covering, forcing the student to put probability everywhere the teacher does. With teacher-level capacity that would be fine, but a 0.5B student cannot represent every mode of a 7B model's next-token distribution, so "cover all the modes" makes it smear its limited mass thin — including over the teacher's low-probability tail — and at free run it samples incoherent, hedge-everything continuations. The other prior line, RL fine-tuning against an outcome reward, fixes the state mismatch by sampling on-policy but gives only a sparse scalar per episode, roughly $O(1)$ bits where a per-token target carries $O(N)$, so credit assignment is expensive and it needs a verifier or a hackable learned reward. What we need is a signal that is simultaneously on-policy, dense at every token, and mode-seeking under capacity mismatch.

I propose On-Policy Distillation (OPD): train the student on its own rollouts under a dense per-token signal, the per-token reverse KL between the student and teacher next-token distributions, summed over the completion tokens. Start from the target the two failures jointly select — the sequence-level reverse KL under student rollouts,
$$L(\theta) = \mathbb{E}_x\, D_{KL}\!\big(p_S(\cdot|x)\,\|\,p_T(\cdot|x)\big) = \mathbb{E}_x\,\mathbb{E}_{y\sim p_S(\cdot|x)}\big[\log p_S(y|x) - \log p_T(y|x)\big].$$
Two design choices are forced here. Sampling $y\sim p_S$ makes it on-policy, so the expectation is over exactly the states the student visits at inference and the compounding-error cascade is gone by construction — a weird state the student drifts into now appears in training with a label. And the *reverse* direction, weighted by $p_S$ rather than $p_T$, is mode-seeking: $D_{KL}(p_S\|p_T)=\sum_v p_S(v)\log\!\big(p_S(v)/p_T(v)\big)$ is large exactly where the student places mass the teacher dislikes, so to drive it down the student *withdraws* mass from any token the teacher finds unlikely and concentrates on one coherent teacher mode. Fitting one Gaussian to a two-bump mixture makes the contrast concrete: forward KL parks the Gaussian in the valley to cover both bumps (the incoherent average), reverse KL snaps onto one bump and commits. Committing to one executable behavior is the right instinct when the student is the smaller party. Reverse KL is also unhackable — the "reward" is the teacher's own log-prob of the student's token, so low KL always means high-teacher-probability behavior, with no separate reward model to game — and it is dense, $O(N)$ bits for an $N$-token sequence.

The difficulty is that, unlike forward KL, reverse KL is an expectation under the student's own distribution — an RL problem. The textbook policy gradient gives
$$\nabla L = -\,\mathbb{E}_{y\sim p_S}\sum_t (R_t - 1)\,\nabla \log p_S(y_t|y_{<t},x),\qquad r_t = \log\frac{p_T(y_t|\cdot)}{p_S(y_t|\cdot)},\quad R_t = \sum_{t'\ge t} r_{t'},$$
where $r_t$ is the per-step reward, $R_t$ the reward-to-go, and the $-1$ is the normalization piece from differentiating $\log p_S$ inside the expectation. Running this directly is a mess, and I can name why before running it: $R_t$ sums a stochastic future, so the estimator is high-variance on long sequences; a small student discovers degenerate strings (a phrase repeated) that the smooth teacher scores highly — reward hacking; and because each $r_t\approx\log(p_T/p_S)$ summed over many tokens makes $R_t$ more negative the longer the sequence, the objective mechanically prefers short or empty responses. Taming all this needs baselines, teacher-mixed sampling with importance weights, length normalization, and PPO clipping — exactly the machinery that drags us away from the stable supervised loop. The crucial move is to look at *where* the mess comes from. Decompose the gradient per step into the immediate term and everything after:
$$\nabla L = -\,\mathbb{E}_{y\sim p_S}\sum_t \nabla\,\mathbb{E}_{y_t\sim p_S(\cdot|y_{<t},x)}[r_t] \;-\; \mathbb{E}_{y\sim p_S}\sum_t R_{t+1}\,\nabla \log p_S(y_t|\cdot) \;=\; (\nabla L)_{\text{Single}} + (\nabla L)_{\text{Long}}.$$
Now stare at the inner object of the Single term. It is a sum over the *whole vocabulary*,
$$\mathbb{E}_{y_t\sim p_S(\cdot|y_{<t},x)}[r_t] = \sum_v p_S(v|y_{<t},x)\,\log\frac{p_T(v|\cdot)}{p_S(v|\cdot)} = -\,D_{KL}\!\big(p_S(\cdot|y_{<t},x)\,\|\,p_T(\cdot|y_{<t},x)\big),$$
the per-token reverse KL evaluated analytically at the prefix $y_{<t}$. With the full student and teacher logits at that position in hand, this is computable exactly by summing over $V$ and is differentiable in $\theta$ — zero sampling noise. Every pathology — the variance, the reward hacking via sampled degenerate completions, the reward-to-go length bias — lives instead in $(\nabla L)_{\text{Long}}$, the term with $R_{t+1}$ and the REINFORCE factor and the sampled future. So the question sharpens to: do I need the Long term at all? It buys long-horizon credit assignment, which is essential when reward is *sparse* and you only learn at the end. But my reward is dense — the teacher tells me at token $t$, right now, how good token $t$ was. The whole justification for tolerating long-horizon variance is absent. So I set the discount factor to zero and drop $(\nabla L)_{\text{Long}}$ entirely. This is a deliberate surrogate, not a change of objective: by the auto-regressive factorization $D_{KL}(p_S(\cdot|x)\|p_T(\cdot|x)) = \mathbb{E}_{y\sim p_S}\sum_t[\log p_S(y_t|\cdot)-\log p_T(y_t|\cdot)]$, the sequence KL is the expectation of nonnegative conditional KLs under the student's prefix distribution, and dropping the Long term keeps the exact gradient of those conditional-KL factors on student-visited states while discarding only the future-prefix score-function part. What remains is startlingly simple — at each on-policy state, the analytic per-token reverse KL over the vocabulary, summed over completion tokens and backpropagated — a structurally *supervised* loss that happens to be evaluated on student-generated prefixes. Deleting the Long term sheds every stabilizer at once, since they all existed to tame it.

Three implementation choices are load-bearing and easy to get subtly wrong. Temperature: divide *both* logit tensors by the same distillation temperature before the softmax, so the divergence sees the teacher's relative preferences among plausible tokens rather than its argmax; it must hit both distributions symmetrically, or the KL would be measuring a sharpness mismatch instead of behavior, and a training temperature near 1 keeps the rollouts diverse enough to visit a real spread of the student's own states. Masking: the loss is meaningful only on completion tokens, so reduce over positions where $\texttt{labels}\neq-100$ and never admit prompt or padding. Reduction: "batchmean" means a *per-token* mean — the sum of per-token KLs divided by the number of unmasked tokens, with the denominator clamped to at least 1 — which makes the loss length-scale-free, the same length-bias hygiene I got for free by dropping $R_t$, now made explicit. And the genuinely fiddly point is the KL direction. The framework primitive computes $\sum p_{\text{target}}(\log p_{\text{target}} - \text{input})$, which is $D_{KL}(p\|q)$ when $\text{input}=\log q$ and $\text{target}=\log p$. I want $D_{KL}(p_S\|p_T)$, so the *student* log-probs are the target and the *teacher* log-probs are the input: $\texttt{kl\_div(input}=\text{teacher\_log\_probs},\ \texttt{target}=\text{student\_log\_probs},\ \texttt{log\_target=True)}$. The off-the-shelf default with the arguments swapped computes forward KL — precisely the mass-covering objective I am trying to escape — so it is worth stating out loud. Seen from above, OPD is the corner $\lambda=1$ (fully on-policy), divergence $=$ reverse KL (the $\beta=1$ endpoint of the generalized JSD family $D_{JSD}(\beta)=\beta D_{KL}(P\|M)+(1-\beta)D_{KL}(Q\|M)$, $M=\beta P+(1-\beta)Q$, where $\beta\to0$ behaves like forward KL) of the general objective $(1-\lambda)\,\mathbb{E}_{\text{data}}[D(p_T\|p_S)] + \lambda\,\mathbb{E}_{x,\,y\sim p_S}[D(p_T\|p_S)]$ with no backprop through sampling — on-policy because the cascade demanded it, reverse KL because capacity mismatch demanded mode-seeking, analytic-and-discount-zero because the teacher's dense log-probs made the closed form available.

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(
    student_logits: torch.Tensor,    # [B, T, V] student logits over the rollout tokens
    teacher_logits: torch.Tensor,    # [B, T, V] frozen-teacher logits over the same tokens
    labels: torch.Tensor = None,     # [B, T]; -100 on prompt/padding, token id on completion
    beta: float = 1.0,               # reverse-KL endpoint; kept for trainer API
    temperature: float = 1.0,
    reduction: str = "batchmean",
    step: int = 0,
    total_steps: int = 0,
    lmbda: float = 1.0,              # student-rollout corner; applied upstream by the trainer
) -> torch.Tensor:
    # On-Policy Distillation: per-token reverse KL on the student's own tokens, KL(p_S || p_T).
    # Mode-seeking under capacity mismatch; dense at every token; discount factor 0
    # (no long-horizon term) makes it a supervised-style differentiable loss.

    # Shared distillation temperature: soften BOTH distributions equally so the divergence
    # measures behavior, not a sharpness mismatch.
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_S || p_T) = Sum_v p_S(v) [log p_S(v) - log p_T(v)], summed over vocab -> [B, T].
    # F.kl_div(input=log_q, target=log_p, log_target=True) = Sum_v p_target (log p_target - input).
    # input = teacher_log_probs, target = student_log_probs  ==>  KL(p_S || p_T) (reverse).
    # Swapping the two args yields forward KL(p_T || p_S) -- the wrong direction.
    per_token = F.kl_div(
        teacher_log_probs, student_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)  # [B, T]

    # Only completion tokens count; mask out prompt/padding (labels == -100).
    if labels is not None:
        mask = labels != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(
            per_token.numel(), device=per_token.device, dtype=torch.long
        ).clamp_min(1)

    if reduction == "batchmean":          # per-token mean -> length-scale-free
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```

For when the trainer wants the sampled-token advantage instead of the analytic loss, the score-function realization of the same discount-0 step keeps only the immediate token and includes the $-1$ normalization term from differentiating the student's own $\log p_S$ inside the reverse-KL expectation:

```python
reverse_kl = sampled_logprobs - teacher_logprobs   # log p_S(y_t) - log p_T(y_t)
r_t = -reverse_kl                                  # log p_T(y_t) - log p_S(y_t)
pg_weight = r_t - 1.0                              # exact score-function weight, gamma = 0
# loss_pg = -(pg_weight.detach() * sampled_logprobs), masked to completion tokens.
# Prefer the analytic vocabulary-sum loss above when teacher logits are available.
```
