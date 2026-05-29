OK, let me think this through from scratch. I have a pile of human preference data — for each prompt x, a pair of completions where a person said "this one is better than that one," written y_w ≻ y_l. I have a pre-trained, supervised-fine-tuned language model π_SFT that's fluent but doesn't reliably produce the responses people actually want. I want to bend it toward the preferences. The whole field has a recipe for this, and the recipe works, but it hurts to run. Let me first be precise about what the recipe is actually optimizing, because if I understand the *objective* cleanly I might be able to reach it by a shorter road than the one everyone takes.

The objective everyone writes down is: fit a reward model to the preferences, then push the policy to earn high reward while staying near the reference. Formally,

  max_π  E_{x~D, y~π(·|x)} [ r(x,y) ] − β · KL( π(y|x) ‖ π_ref(y|x) ).

The reward part wants the policy to chase whatever the humans liked; the KL part is a leash that keeps it near π_ref. The leash matters for real reasons: the reward model is only accurate near the data it was trained on, so I don't want the policy to wander off into regions where the reward is garbage; and without the leash the policy mode-collapses onto a few strings that happen to score high, killing diversity. β sets the leash length.

And where does the reward come from? From the preference data, through a preference model. The standard choice is Bradley-Terry: a latent reward r*(x,y) explains the human choices via

  p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / ( exp(r*(x,y1)) + exp(r*(x,y2)) ).

Divide top and bottom by exp(r*(x,y1)) and that's just σ(r*(x,y1) − r*(x,y2)), the logistic of the reward *difference*. So I fit r_φ by maximum likelihood on the pairs — minimize −E_{(x,y_w,y_l)} [ log σ( r_φ(x,y_w) − r_φ(x,y_l) ) ]. That's a plain binary classification loss: is the preferred completion scored higher than the dispreferred one? Fine. That stage is cheap and stable; it's logistic regression.

The pain is the *next* stage. To actually maximize that KL-constrained objective over a discrete, autoregressive language model, the gradient of E_{y~π}[r] with respect to π's parameters isn't something you can just backprop — y is sampled, the generation is discrete. So everyone reaches for reinforcement learning: PPO. And PPO here means a separate reward model in memory, a separate value/critic network, *sampling from the multi-billion-parameter policy in the inner loop of training*, advantage estimation, ratio clipping, reward normalization with some baseline, and a KL coefficient you have to babysit. It's slow, it's finicky, and it's unstable. People paper over the instability with tricks — normalizing rewards by a human-completion baseline, for instance, which is really a one-sample Monte-Carlo estimate of some normalizing quantity.

So here's the itch. The reward-fitting stage is a clean supervised classification problem. The policy-fitting stage is a fragile RL problem. Both stages are, in the end, trying to satisfy the same preference data. Why are there two stages, and why is the second one RL at all? Could I collapse them — go straight from preferences to a policy with one supervised loss?

Let me look at the closed form of the optimum, because if I'm going to skip the RL I need to know what RL is even aiming at. The KL-constrained objective is actually solvable in closed form for an unconstrained (non-parametric) policy. Let me actually do it, I don't want to hand-wave the one fact I'm going to build everything on.

Take the objective, write the KL out as an expectation:

  max_π E_{x} E_{y~π(·|x)} [ r(x,y) − β log( π(y|x) / π_ref(y|x) ) ].

Pull out a −β and flip max to min:

  min_π E_{x} E_{y~π(·|x)} [ log( π(y|x) / π_ref(y|x) ) − (1/β) r(x,y) ].

Now I want to fold that reward term into the log so the whole bracket looks like a single log-ratio — then it might be a KL and I can use the fact that KL ≥ 0. The reward inside the log would need to appear as exp(r/β). So define

  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y)/β ),    Z(x) = Σ_{y'} π_ref(y'|x) exp( r(x,y')/β ).

Z(x) is just whatever it takes to make π* sum to one over y; it's a function of x and π_ref and r, but not of π. Now rewrite the bracket. We have −(1/β) r(x,y) = −log exp(r/β) = −log( Z(x) π*(y|x) / π_ref(y|x) ) = −log( π*(y|x)/π_ref(y|x) ) − log Z(x). Substitute:

  log( π/π_ref ) − (1/β) r = log( π/π_ref ) − log( π*/π_ref ) − log Z(x) = log( π/π* ) − log Z(x).

So the objective is

  min_π E_x [ E_{y~π} [ log( π(y|x)/π*(y|x) ) ] − log Z(x) ] = min_π E_x [ KL( π(·|x) ‖ π*(·|x) ) − log Z(x) ].

Z(x) doesn't depend on π, so it's a constant for the minimization. The only term I control is the KL, which by Gibbs' inequality is ≥ 0 with equality exactly when π = π*. So the optimum is

  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y)/β ).

Good — and that completion-of-the-square trick (manufacture π* so the reward term becomes part of a KL) is the whole derivation. The optimal policy is the reference, reweighted by exp(reward/β). Tilt probability mass toward high-reward completions, and the tilt sharpens as β shrinks — β is the weight on the KL leash, so large β keeps the policy pinned near π_ref while small β lets it chase reward hard. Either way the *form* is an exponential tilt of π_ref.

Now — this is exactly the place where prior work gets stuck, and where I almost get stuck too. This closed form is famous; reward-weighted regression (Peters & Schaal 2007), advantage-weighted regression (Peng 2019), the distributional-control work — they all know the optimum is this exp-tilted distribution. But they can't *use* it directly, because Z(x) = Σ_{y'} π_ref(y'|x) exp(r(x,y')/β) is a sum over all possible completions. For language, that's all possible sequences — combinatorially many. You can't compute Z, you can't even cheaply estimate it (importance sampling over sequences from an LM is brutally expensive). So the closed form gets demoted to a *target you approximate*: weight your samples by exp(r/β) and regress, or do RL toward it. The intractable Z is the wall everyone hits, and it's why we're back to sampling-based optimization.

Let me stare at this wall for a second instead of going around it the usual way. The form is π*(y|x) = π_ref(y|x) exp(r/β) / Z(x). I keep reading it left-to-right: "given a reward r, here is the optimal policy, but Z is intractable." What if I read it right-to-left? I have this equation relating r, π*, π_ref, and Z. It's one equation. I've been treating r as known and solving for π*. But I could just as well treat π* as the object and *solve for r*. Take logs:

  log π*(y|x) = log π_ref(y|x) + (1/β) r(x,y) − log Z(x).

Solve for r:

  r(x,y) = β log( π*(y|x) / π_ref(y|x) ) + β log Z(x).

Interesting. This says: *any* reward function r can be written in terms of its own optimal policy π_r, the reference π_ref, and an x-only term β log Z(x). The reward and the optimal policy are two views of the same object; given one I have the other. So instead of "fit r, then derive π," I could parameterize the *policy* directly and read off the reward it implies as r̂(x,y) = β log( π(y|x)/π_ref(y|x) ) + β log Z(x). The policy network would secretly *be* a reward model.

But Z(x) is still sitting right there in the expression for r. I haven't escaped it; I've just moved it. If I want to fit this thing to preference data I'll have to evaluate r, and r contains β log Z(x). Dead end?

No — wait. Where does r enter the preference data? Through Bradley-Terry. And Bradley-Terry depends only on the *difference* r(x,y1) − r(x,y2). Look at what β log Z(x) is: a function of x *only*. It's the same for y1 and for y2, because both completions share the prompt x. When I take the difference, it cancels. Let me actually substitute and watch it die. Plug r*(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x) into the Bradley-Terry expression:

  p*(y1 ≻ y2 | x)
   = exp( β log(π*(y1)/π_ref(y1)) + β log Z(x) )
     / [ exp( β log(π*(y1)/π_ref(y1)) + β log Z(x) ) + exp( β log(π*(y2)/π_ref(y2)) + β log Z(x) ) ].

Factor exp(β log Z(x)) out of the numerator and out of both denominator terms, and it cancels:

   = exp( β log(π*(y1)/π_ref(y1)) ) / [ exp( β log(π*(y1)/π_ref(y1)) ) + exp( β log(π*(y2)/π_ref(y2)) ) ].

Divide top and bottom by the numerator:

   = 1 / ( 1 + exp( β log(π*(y2)/π_ref(y2)) − β log(π*(y1)/π_ref(y1)) ) )
   = σ( β log(π*(y1)/π_ref(y1)) − β log(π*(y2)/π_ref(y2)) ).

The partition function is gone. Completely. The preference probability under the *true* reward's optimal policy is a logistic function of the difference of β·log-ratios between the policy and the reference. No Z, no sum over sequences, nothing intractable. The thing that made the closed form unusable evaporated the moment I expressed preferences through it, because preferences only ever see reward *differences* and Z was a reward *offset*.

This reframes the implicit reward as r̂(x,y) = β log( π(y|x)/π_ref(y|x) ) — the β log Z(x) piece doesn't matter for anything preferences can observe, so I can just drop it and define the policy's implicit reward as the log-ratio. The policy is secretly a reward model, and the secret reward is β times how much more (or less) likely the policy makes a completion relative to the reference.

Now I have the optimal policy satisfying a Bradley-Terry model whose reward is the implicit log-ratio. So I flip the entire pipeline. Prior work fits r by MLE on the preference likelihood, then solves for π. I'll write the same preference likelihood but with r already substituted by the implicit reward, and fit π by MLE *directly*. Parameterize π_θ, and the negative log-likelihood of the preferences is

  L_DPO(π_θ; π_ref) = − E_{(x,y_w,y_l)~D} [ log σ( β log( π_θ(y_w|x)/π_ref(y_w|x) ) − β log( π_θ(y_l|x)/π_ref(y_l|x) ) ) ].

That's it. That's the whole objective. It's structurally identical to the reward-model classification loss −E[log σ(r(x,y_w) − r(x,y_l))] — same binary cross-entropy — except r has been replaced by the implicit reward β log(π_θ/π_ref). One stage. A supervised classification loss on the policy. No separate reward model. No value function. No sampling from the policy during training — y_w and y_l come straight from the fixed dataset. No RL. The reward model and the policy are the *same network*, viewed two ways.

Let me make sure I believe two things before I get attached to this. First: did I lose generality by insisting the reward have the form β log(π/π_ref)? It looks like a restriction — I've forced rewards to be log-ratios of distributions. Second: is the thing I'm fitting actually consistent — does minimizing this loss really recover the optimal policy of the original problem, or just something that looks like it?

For the generality worry. Notice the reward is only ever identified up to an x-only shift anyway — that's the Bradley-Terry under-specification: r and r + g(x) give identical preference probabilities, because the g(x) cancels in every difference. So rewards naturally come in equivalence classes, where r ~ r' iff r(x,y) − r'(x,y) = f(x) for some f. Let me check this equivalence is exactly the right one — that it doesn't matter *which* representative of a class I land on. Two things to verify. (a) Members of a class induce the same preferences. Take r'(x,y) = r(x,y) + f(x). Under Bradley-Terry (and Plackett-Luce generally), each factor is exp(r')/Σ exp(r'); the exp(f(x)) factors out of numerator and the whole denominator sum and cancels, leaving exactly the r-induced distribution. So preferences can't tell class members apart. (b) Members of a class induce the same *optimal policy*. From π_r(y|x) = π_ref(y|x) exp(r(x,y)/β) / Σ_{y'} π_ref(y'|x) exp(r(x,y')/β): replace r by r + f(x); the exp(f(x)/β) appears once in the numerator and once in every term of the normalizing sum, so it cancels top and bottom, and π_{r'} = π_r. So the policy can't tell them apart either. Good — the class is exactly the resolution at which everything I care about (preferences and the optimal policy) is determined, and I only need to recover *some* representative of the right class.

Now the generality claim sharpens into something provable: does the reparameterization r(x,y) = β log(π(y|x)/π_ref(y|x)) hit *every* equivalence class? Take any reward r at all. It induces an optimal policy π_r by the closed form. I showed r(x,y) = β log(π_r/π_ref) + β log Z(x). Define a projection that subtracts the x-only piece:

  f(r; π_ref, β)(x,y) = r(x,y) − β log Σ_{y'} π_ref(y'|x) exp(r(x,y')/β) = r(x,y) − β log Z(x).

Since β log Z(x) depends only on x, this projected reward is in the same equivalence class as r. And by the relation above, the projected reward equals exactly β log(π_r(y|x)/π_ref(y|x)) — it's in the desired form. So every class contains a representative of the reparameterized form; the reparameterization loses no generality. (And uniquely so: suppose two policies π and π' give class-equivalent reparameterized rewards, with β log(π(y|x)/π_ref(y|x)) = β log(π'(y|x)/π_ref(y|x)) + f(x). Then π(y|x) = exp(f(x)/β)π'(y|x); sum over y, both are distributions so 1 = exp(f(x)/β), so f ≡ 0 and π = π'. Each class has exactly one reparameterized representative.) Another way to see what the reparameterization *picks* out of each class: the unique r whose induced optimal policy already normalizes, Σ_{y'} π_ref(y'|x) exp(r(x,y')/β) = 1, i.e. the representative for which Z(x) = 1 and the optimal policy is analytically just the tilt with no leftover normalizer to compute. That's the whole trick — constrain the under-specified reward family precisely enough that the optimal policy becomes tractable for every x, without shrinking the family.

And consistency: since the loss is literally a Bradley-Terry MLE in a faithful reparameterization, it inherits the consistency properties of Bradley-Terry maximum likelihood under the usual assumptions on the preference distribution (Bong & Rukhin 2022). I'm not fitting an approximation to the objective; I'm fitting the same statistical model in different coordinates.

Let me now look at what the gradient does, because I want to understand the *mechanism*, and because there's a naive cousin of this objective that I suspect fails, and I want to see why. Differentiate L_DPO. Let s = β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)) be the argument of σ — the preferred-minus-dispreferred implicit-reward difference. The per-example loss is −log σ(s). So d/dθ [−log σ(s)] = −(σ'(s)/σ(s)) ∇s. Using σ'(s) = σ(s)(1−σ(s)) and 1−σ(s) = σ(−s), the prefactor σ'(s)/σ(s) collapses to 1 − σ(s) = σ(−s). And ∇s = β[ ∇log π_θ(y_w) − ∇log π_θ(y_l) ], since π_ref is constant in θ. So

  ∇_θ L_DPO = − E_{(x,y_w,y_l)} [ β · σ( −s ) · ( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) ) ].

Now read −s back in reward terms. s = r̂(x,y_w) − r̂(x,y_l) with r̂ = β log(π_θ/π_ref). So −s = r̂(x,y_l) − r̂(x,y_w), and the gradient is

  ∇_θ L_DPO = − β · E [ σ( r̂(x,y_l) − r̂(x,y_w) ) · ( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) ) ].

Look at the two pieces. The bracket ∇log π_θ(y_w) − ∇log π_θ(y_l): a gradient step against this *raises* the log-probability of the preferred completion and *lowers* that of the dispreferred one. Exactly what I want. The scalar weight σ(r̂(x,y_l) − r̂(x,y_w)): this is large (→1) precisely when the implicit reward orders the pair *wrong* — when it scores the dispreferred completion *higher* than the preferred one — and small (→0) when the pair is already ordered correctly with margin. So the update is self-pacing: it pours gradient into the examples the model currently gets wrong, and stops pushing on the ones it already has right.

That weight is the crux, and it's what separates this from the naive thing. The naive objective is "just raise log p(y_w) and lower log p(y_l)" — unweighted, which is the unlikelihood baseline. That's exactly the bracket ∇log π_θ(y_w) − ∇log π_θ(y_l) with weight fixed at 1, no σ term, no reference ratios. With no brake on the likelihood *minimization* of y_l, the model keeps driving down probabilities without bound and degenerates — it collapses into repetitive non-text (a TL;DR that outputs "when when when when …"). The σ(r̂_l − r̂_w) factor is what prevents this: once the preferred completion is comfortably ahead, the weight vanishes and the push stops, and the whole thing is anchored to π_ref through the log-ratio so "lowering p(y_l)" is measured relative to the reference rather than in absolute terms. The dynamic, per-example weight, scaled by β to respect the strength of the KL constraint, is doing real work.

There's a generalization sitting here too, almost for free. Bradley-Terry was the K=2 case of Plackett-Luce, which handles full rankings of K items: the probability of a ranking τ is the product over ranks of exp(r(y_{τ(k)})) / Σ_{j≥k} exp(r(y_{τ(j)})). The same cancellation works — each factor is a softmax over a subset of completions all sharing the prompt x, so substituting r = β log(π/π_ref) + β log Z(x) makes the β log Z(x) cancel within every factor. So if I ever have ranked data instead of pairs, the objective is the obvious log-product analogue with the implicit log-ratio reward. Nice to know it's not special to pairs.

One more thing I want to understand, because it tells me *why* the RL route was unstable in the first place and reassures me I'm not just trading one instability for another. Go back to the actor-critic view of stage 3. If a fixed reward r_φ induces π*(y|x) = π_ref(y|x)exp(r_φ(x,y)/β)/Z_φ(x), then minimizing KL(π_θ ‖ π*) is, after multiplying by −β, equivalent to maximizing E_{π_θ}[r_φ(x,y) − β log(π_θ(y|x)/π_ref(y|x)) − β log Z_φ(x)]. The subtracted term V_φ(x) = β log Z_φ(x) is the soft value of the reference distribution under r_φ. It is an x-only baseline, so it does not change the expected policy-gradient direction, but it is exactly the baseline that would reduce variance if I were estimating the gradient with samples. Standard actor-critic has to learn a value baseline for this role, or hack around it with a reference-completion estimate of the normalizing term. The reparameterization I derived makes this term *cancel structurally* — it never has to be estimated at all, because preferences only see differences. So the stability isn't luck; it's the same cancellation that removed Z, viewed from the optimization side.

Alright. I'm confident. Let me write the code, and let me make sure it mirrors how this actually gets implemented rather than a toy. The objective needs, per preference pair, four sequence-level log-probabilities: policy-on-chosen, policy-on-rejected, reference-on-chosen, reference-on-rejected. A "sequence log-probability" is the sum over completion tokens of log p(token | prefix) — sum, not average, because that's the log-likelihood the Bradley-Terry derivation is written in. The autoregressive shift matters: the logit at position t predicts the label at t+1, and prompt or padding positions must be masked out before the sum. The reference is frozen, so its log-probs are computed under no_grad, and chosen plus rejected completions can be concatenated into one forward pass per model. Then the loss is just −logsigmoid(β · ((policy_chosen − policy_rejected) − (reference_chosen − reference_rejected))), which is exactly −log σ(β(log-ratio_w − log-ratio_l)) once I group the chosen/rejected and policy/reference terms. If I want conservative labels, I can smooth the binary target by mixing in the reversed comparison; if I want the reference-free ablation, I set the reference log-ratio to zero. The implicit rewards β(policy − reference), detached, are worth logging to watch the margin grow.

```python
import torch
import torch.nn.functional as F

def get_batch_logps(logits, labels, average_log_prob=False):
    # logits: (B, T, V) from the LM; labels: (B, T), with -100 on prompt/pad.
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()             # token t is predicted from positions < t
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)               # only score completion tokens
    labels[labels == -100] = 0                 # gather needs valid indices
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    summed_logps = (per_token_logps * loss_mask).sum(-1)
    if average_log_prob:                       # default False: we want the SUM
        return summed_logps / loss_mask.sum(-1)
    return summed_logps                        # sequence log-prob, shape (B,)

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta, label_smoothing=0.0, reference_free=False):
    pi_logratios  = policy_chosen_logps   - policy_rejected_logps      # log π_θ(y_w)/π_θ(y_l)
    ref_logratios = reference_chosen_logps - reference_rejected_logps  # log π_ref(y_w)/π_ref(y_l)
    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)
    logits = pi_logratios - ref_logratios     # = (r̂_w - r̂_l)/β ; β log Z cancelled here
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        -F.logsigmoid(-beta * logits) * label_smoothing
    )
    chosen_rewards   = beta * (policy_chosen_logps   - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards

def concatenated_forward(model, batch):
    all_logits = model(batch["concat_input_ids"],
                       attention_mask=batch["concat_attention_mask"]).logits
    all_logps = get_batch_logps(all_logits, batch["concat_labels"])
    n_chosen = batch["chosen_input_ids"].shape[0]
    return all_logps[:n_chosen], all_logps[n_chosen:]

def train_step(policy_model, reference_model, batch, beta, optimizer):
    pc, pr = concatenated_forward(policy_model, batch)
    with torch.no_grad():                      # reference is frozen — no gradient, no RL
        rc, rr = concatenated_forward(reference_model, batch)
    losses, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, beta)
    loss = losses.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss.item(), accuracy.item()
```

For knobs: β = 0.1 by default (β = 0.5 for the TL;DR summarization data), the reference is the SFT model, and when there's no SFT model available I make one by fine-tuning on the preferred completions only — π_ref = argmax_π E[log π(y_w|x)] — so that the reference is in-distribution for the preferences I'm fitting, which keeps the implicit reward honest. RMSprop at lr 1e-6 with a short linear warmup over ~150 steps is enough; there's no sampling loop, no critic, no reward model, so there's almost nothing to tune.

The causal chain, start to finish: I want the policy that maximizes preference-reward minus a KL leash to the reference. That objective has a closed-form optimum, an exponential tilt of the reference by the reward, but its partition function is intractable, so nobody uses it directly — they approximate it with RL. Instead of solving the equation for the policy, I solve it for the reward: every reward is β·log(its optimal policy / reference) plus an x-only term. Substituting that into Bradley-Terry, the x-only term — the partition function — cancels, because preferences only see reward differences. So the optimal policy satisfies a Bradley-Terry model whose reward is just β·log(π/π_ref), and I can fit the policy directly by the same maximum-likelihood classification loss used for reward models — no reward model, no value function, no sampling, no RL. The gradient turns out to raise the preferred completion and lower the dispreferred one, weighted by how wrongly the policy currently orders them, which keeps it from degenerating the way an unweighted likelihood objective does. The policy was a reward model all along.
