Let me start from what actually goes wrong when I train one of these big networks. I have far more parameters than I have any right to fit, and even with a large dataset the thing overfits — training accuracy marches toward perfect while held-out accuracy stalls. I reach for the usual tools and notice they all share a shape: weight decay penalizes the weights, dropout perturbs the hidden units, batch norm normalizes activations, early stopping watches the loss. Every one of them reaches into the *insides* of the network. But the thing the network actually hands to the world is the distribution `p_theta(y|x)` that falls out of the softmax, and I'm doing essentially nothing to that. That's a strange blind spot, because the symptom of overfitting I can literally see is in the output: a network that has overfit puts almost all its probability mass on one class. If I train a small fully-connected net on MNIST with dropout and histogram the softmax probabilities over the validation set, the mass piles up at 0 and 1 — the outputs are nearly deterministic spikes. So the disease shows up *in the output distribution*, yet I'm only ever medicating the weights and activations. What would it mean to regularize the output distribution itself?

Before I can answer that I should be honest about why the output is even a good place to act. Two reasons, and they're not cosmetic. First, the output distribution has a natural, fixed scale — it's a probability vector, it sums to one no matter what — whereas the weights don't: the meaning of any single weight depends on every other weight in the network, so a penalty on the weights is entangled with how I happened to parameterize things. A penalty on the output is invariant to the parameterization underneath. Second, there's a subtler point about what the output *contains*. If I think of the network's knowledge as the function from inputs to output distributions rather than as the pile of weights, then the probabilities it assigns to the *wrong* classes are part of that knowledge. Shown a particular car, a net that puts `1e-3` on a similar-looking make and `1e-9` on something unrelated is genuinely better than one that flips those, because the *ratios* among the wrong-class probabilities tell me how the network generalizes — this is the "dark knowledge" that distillation lives off. I'll file that away as a constraint on whatever I build: don't crush the wrong-class ratios.

So, what does "regularize the output distribution" concretely want? The overfit network is *over-confident* — its output is a low-entropy spike. The fix I want is something that discourages those peaked, low-entropy distributions. Entropy is exactly the scalar that measures this. For a softmax output

  `H(p_theta(y|x)) = - sum_i p_theta(y_i|x) log p_theta(y_i|x)`,

it's maximal at the uniform distribution and zero at a one-hot spike. Over-confidence is low entropy; the cure is to push entropy up. And I've seen this move before, just not here. In reinforcement learning, when a policy is a softmax over actions, people add the entropy of the policy to the objective — a `+ beta H(pi)` term — to stop the policy from collapsing onto one action too early, so it keeps exploring. Williams and Peng did this back in 1991 with their entropy-augmented REINFORCE, and it's standard in modern policy-gradient training: the update carries an explicit `beta grad H(pi(s))` term, with `beta` setting how hard you push for stochasticity. The whole point there is to *prevent premature convergence to a deterministic policy*. Stare at that for a second — "premature convergence to a deterministic policy" in RL has the same failure pattern as "an over-confident, low-entropy output distribution" in supervised learning. The RL people add entropy to keep their softmax from spiking. I want to keep *my* softmax from spiking. It's the same term, in a setting where nobody seems to have tried it.

So the candidate is almost embarrassingly direct: take the negative log-likelihood I already minimize and subtract a multiple of the output entropy,

  `L(theta) = - sum log p_theta(y|x) - beta H(p_theta(y|x))`.

Let me make sure the signs say what I mean, because this is the kind of thing that's easy to flip. I *minimize* `L`. The term `- beta H` means that to make `L` small I want `H` large — high entropy, less peaked outputs. Equivalently, I am *penalizing low entropy*, which is penalizing confidence. Good, the sign is right: subtract entropy to penalize confidence. `beta` controls how hard I push.

Now, is this actually a sensible regularizer or am I just going to flatten everything to uniform and destroy the model? Let me look at what the entropy term does to the gradient, because that's where I'll see whether it's surgical or a sledgehammer. I want `partial H / partial z_i`, the gradient of the entropy with respect to the logits `z`, since that's what flows back into the network. I'll need the softmax Jacobian first. With `p_j = exp(z_j) / sum_k exp(z_k)`, differentiate: `partial p_j / partial z_i = p_j (delta_{ij} - p_i)` — the standard result, easy to re-derive by quotient rule (when `i = j` you get `p_j(1 - p_j)`, when `i != j` you get `-p_j p_i`). Now

  `partial H / partial z_i = partial / partial z_i [ - sum_j p_j log p_j ]
                           = - sum_j (partial p_j / partial z_i)(log p_j + 1)`,

where the `+1` is from differentiating the `p_j` outside the log (`partial(p log p)/partial p = log p + 1`). Substitute the Jacobian:

  `= - sum_j p_j (delta_{ij} - p_i)(log p_j + 1)`.

Split it into the `delta_{ij}` piece and the `- p_i` piece:

  `= - p_i (log p_i + 1) + p_i sum_j p_j (log p_j + 1)`.

Now the sum: `sum_j p_j (log p_j + 1) = sum_j p_j log p_j + sum_j p_j = -H + 1`, using `sum_j p_j log p_j = -H` and `sum_j p_j = 1`. So

  `partial H / partial z_i = - p_i log p_i - p_i + p_i(-H + 1) = - p_i log p_i - p_i H = p_i ( - log p_i - H )`.

Let me sit with that result, `partial H / partial z_i = p_i(- log p_i - H)`, because it tells me precisely how surgical this is. The quantity `- log p_i` is the surprisal of class `i`, and `H = sum_i p_i(- log p_i)` is the *mean* surprisal under `p`. So the gradient of the entropy on logit `i` is the deviation of class `i`'s surprisal from the mean, weighted by `p_i` — a "weighted deviation from the mean." Read off the consequences. For the class the model is confident about, `p_i` is large and `- log p_i` is small (below the mean surprisal), so the deviation `(- log p_i - H)` is negative: the entropy term pushes that logit *down*. For a class the model has nearly killed, `p_i ≈ 0`, so even though its surprisal is huge, the whole thing is multiplied by `p_i ≈ 0` and the push is negligible. So this does not yank every dead class up toward uniform indiscriminately. It mostly acts on the dominant class, pulling it down, with the action concentrated wherever `p_i` is non-negligible — exactly the over-confident classes. It flattens the distribution toward uniform but in proportion to how much mass each class already carries, which means it leaves the long tail of nearly-zero wrong classes essentially alone, and therefore preserves their *ratios* — the dark knowledge I told myself not to crush. That's reassuring; the `p_i` weighting is doing real work, not decoration.

Now I have to confront the closest existing thing, because if I can't say why I'm not just reinventing it I haven't understood my own method. Label smoothing replaces the one-hot target `q(k) = delta_{k,y}` with a softened `q'(k) = (1 - epsilon) delta_{k,y} + epsilon u(k)`, `u` uniform. As a loss this is `H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)` — hard-label cross-entropy plus an `epsilon`-weighted `H(u, p)`. And `H(u, p) = - sum_k u(k) log p(k) = D_KL(u || p) + H(u)`, with `H(u)` constant in `theta`. So label smoothing, up to a constant, adds the *forward* KL divergence `D_KL(u || p)` — from the fixed uniform `u` to the model `p`. Interesting. My confidence penalty is adding `- beta H(p)`. Are these the same thing? Let me compute the *reverse* KL, `D_KL(p || u)`, and see:

  `D_KL(p || u) = sum_i p_i log (p_i / u_i) = sum_i p_i log p_i - sum_i p_i log u_i`.

With `u_i = 1/K` constant, `sum_i p_i log u_i = log(1/K) sum_i p_i = - log K`. So

  `D_KL(p || u) = - H(p) + log K`.

There it is. Minimizing `D_KL(p || u)` is, up to the constant `log K`, exactly minimizing `- H(p)`, which is exactly maximizing entropy — my confidence penalty. So the confidence penalty *is* a KL penalty toward uniform, just with the KL direction *reversed* relative to label smoothing. Label smoothing penalizes `D_KL(u || p)`; the confidence penalty penalizes `D_KL(p || u)`. Same target distribution (uniform), opposite direction.

Why should I care which direction? The KL is asymmetric, and the asymmetry is the whole story of how the two regularizers behave differently. Look at what weights the log-ratio in each. In label smoothing's `D_KL(u || p) = sum_i u_i log(u_i / p_i)` the weight on each class is the *constant* `u_i = 1/K`: every class gets equal pressure, fixed, regardless of what the model is doing on it. In the confidence penalty's `D_KL(p || u) = sum_i p_i log(p_i / u_i)` the weight on each class is the model's *own current* `p_i`, which changes as training proceeds and is large precisely on the classes the model is currently confident about. So the confidence penalty adaptively concentrates its pressure on the over-confident predictions and ignores classes the model already assigns little mass — which is exactly the behavior I just read off the gradient `p_i(- log p_i - H)`. The two pictures agree, and that's the point of the KL-direction observation: it's not a curiosity, it's the formal reason the confidence penalty is adaptive where label smoothing is uniform.

The reversed direction buys me two more things I wanted from the start. First, label smoothing forces every incorrect class toward the same target value `epsilon/K`, which flattens the dark-knowledge ratios; the confidence penalty never specifies a target for the wrong classes at all — it only asks the whole distribution to be less peaked — so the relative sizes among the small probabilities survive. Second, label smoothing presupposes a prior `u`. Uniform is fine when the classes are balanced, but think about language modeling, where the words follow a brutally non-uniform frequency distribution: choosing `u` becomes a real and arbitrary decision — uniform? unigram? some higher-order n-gram? The confidence penalty as I've written it, `- beta H(p)`, makes *no* commitment about wrong-class targets and so dodges that decision entirely. Seeing the penalty as `D_KL(p || u)` with `u` uniform also tells me I could swap in another reference distribution if a task truly supplied one; the clean version stays with entropy and avoids that choice.

Let me push on the mechanism a little more, because I have a second observation I want to explain rather than just assert. With one-hot targets and softmax, the per-logit cross-entropy gradient is `partial ell / partial z_k = p(k) - q(k)`, bounded in `[-1, 1]`. Now picture a *misclassified* training example where the model is confidently wrong: it has put `p` near 1 on some wrong class and `p` near 0 on the true class. The true class has `q = 1` and `p ≈ 0`, so its gradient component is `p - q ≈ -1` in magnitude — maximal — and the runaway wrong class contributes near `+1`. A confident, peaked output on a misclassified example therefore produces a large gradient. So if a regularizer keeps the outputs from peaking, it should keep the gradient norms smaller and steadier through training. I don't need to run anything to believe this — it falls straight out of the `p - q` form. It's a nice consistency check: the same mechanism (don't let the output spike) that fixes over-confidence also tames the gradients that over-confidence on hard examples produces.

Now a real wall, and it comes from the RL analogy I started with. In RL the entropy bonus is something you want *on throughout* training — you keep exploring, you don't want to converge early at all. But supervised learning is different: I actually *do* want to converge fast. I want the network to commit to the right answers quickly on the easy examples; it's only the over-confidence *near the end* of training, once it starts memorizing, that I want to fight. A constant `beta H` from step one is a blunt instrument — early on it's fighting the very convergence I want, late on it's doing the job I hired it for. So the schedule matters: I want the confidence penalty *weak at the start and strong near convergence*. The simplest realization is to anneal `beta` — ramp it up over training — which directly encodes "let it converge first, then keep it humble."

There's a second, sharper way to get "strong only near convergence" that doesn't require me to guess a schedule, and it falls out of thinking about *when* I actually want the penalty active. I only want to push back when the output has *already become too confident* — when its entropy has dropped below some acceptable level. So instead of rewarding entropy always, reward it only up to a threshold `Gamma`, or equivalently charge the loss only for the entropy shortfall below `Gamma`. A hinge does that, but I have to get the sign right. If I put the shortfall term into the loss with a negative sign, then below the threshold the extra term behaves like a constant plus `+ beta H`; minimizing would push `H` downward, exactly backward. The loss-side hinge has to be positive:

  `L(theta) = - sum log p_theta(y|x) + beta max(0, Gamma - H(p_theta(y|x)))`.

When `H >= Gamma` the output is humble enough and `max(0, Gamma - H) = 0`, no penalty, no interference with convergence. When `H < Gamma` the output has gotten too peaked and the penalty switches on with strength proportional to how far below the threshold it is; the derivative with respect to `H` is `-beta`, so minimizing pushes entropy upward. Up to the constant `beta Gamma`, this is the same as adding `- beta min(H, Gamma)`: the negative-entropy reward is clipped once the entropy is high enough. The cost is one extra hyperparameter, `Gamma`, so the single-knob `- beta H` form remains the simplest default.

So where does `beta` itself sit? It's the one knob, and its right value depends on the task, which makes sense: it trades the data-fitting term against the humility term, and how much humility helps depends on how badly the model overfits and how peaked its natural outputs are. I won't pretend there's a universal constant; the knob is genuinely task-dependent. The practical selling point is that I can leave the model's *other* hyperparameters untouched and sweep this one regularization strength.

Let me also sanity-check that this thing is cheap, because a regularizer that doubles training cost isn't a drop-in. The entropy `H(p) = - sum_i p_i log p_i` is one extra reduction over the logits I already computed for the cross-entropy; its gradient `p_i(- log p_i - H)` is closed-form, no extra forward or backward passes, no auxiliary network. That's the contrast with something like virtual adversarial training, which needs multiple extra forward/backward passes per step to estimate its smoothness gradient — fine in principle but expensive to grid-search and train. The confidence penalty adds essentially nothing to the step cost.

Now let me write the actual term, the thing that drops into the fixed training loop's empty regularizer slot. I have logits `outputs` of shape `[B, K]` and I want the mean negative entropy times `beta`. The one numerical subtlety: I should not form the softmax and then take its log separately, because `log` of a softmax that has small entries amplifies floating-point error; the stable route is `log_softmax`, computed in one pass, and then recover `p = exp(log p)`. The entropy of each row is `- sum_i p_i log p_i`, average over the batch, and I return *negative* that entropy scaled by `beta` so that minimizing the loss maximizes entropy:

```python
import torch
import torch.nn.functional as F


def compute_regularization(model, inputs, outputs, targets, config):
    """Confidence penalty: subtract beta * H(p) from the loss, penalizing
    low-entropy (over-confident) softmax outputs.

      L = cross_entropy  +  compute_regularization
        = - sum log p_theta(y|x)  -  beta * H(p_theta(y|x))
    """
    beta = float(config.get("beta", 0.1))        # single knob, swept on validation data

    # log_softmax once, in one numerically stable pass (no log(softmax) round-trip)
    log_p = F.log_softmax(outputs, dim=-1)       # [B, K], stable log-probabilities
    p = log_p.exp()                              # recover the probabilities

    # H(p) = - sum_i p_i log p_i , averaged over the batch
    entropy = -(p * log_p).sum(dim=-1).mean()

    # add the NEGATIVE entropy => minimizing the loss maximizes entropy
    # => penalizes confident (low-entropy) predictions
    return -beta * entropy
```

And the sharper, thresholded variant, the same thing behind a hinge so the penalty only switches on once the output has dropped below the entropy threshold `Gamma`:

```python
def compute_regularization_thresholded(model, inputs, outputs, targets, config):
    """Hinge confidence penalty: + beta * max(0, Gamma - H(p)).
    Penalize only when the output entropy has fallen below Gamma -- i.e. only
    once the model has become too confident. Weak early, strong near convergence,
    without an explicit beta schedule."""
    beta = float(config.get("beta", 0.1))
    Gamma = float(config.get("entropy_threshold", 0.8))
    log_p = F.log_softmax(outputs, dim=-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(dim=-1).mean()
    return beta * torch.clamp(Gamma - entropy, min=0.0)
```

Let me retrace the whole chain to make sure it actually closes. I started stuck: big networks overfit, the symptom is an over-confident, low-entropy output distribution, yet every standard regularizer acts on weights or activations and leaves the output alone — and the output is the right place to act because it has a natural scale (parameterization-invariant) and its wrong-class ratios carry generalization information I shouldn't destroy. Entropy is the scalar that measures over-confidence, and I recognized the cure as the entropy bonus that RL already uses to stop a policy from collapsing onto a deterministic action — the same "don't let the softmax spike" move, in a setting where it had not been tried. That gave the objective `- sum log p - beta H(p)`, with the sign chosen so minimizing the loss maximizes entropy. Differentiating the entropy through the softmax Jacobian gave `partial H / partial z_i = p_i(- log p_i - H)`, a weighted deviation from the mean surprisal, which showed the penalty acts mostly on the confidently-predicted classes and leaves the near-dead tail (and its ratios) almost untouched — surgical, not a sledgehammer. Computing both KL divergences to uniform revealed `D_KL(p || u) = - H(p) + log K`, so the confidence penalty is label smoothing's penalty with the KL direction reversed; the reversal is exactly why it weights its pressure by the model's own predictions (adaptive, focused on over-confidence) rather than by a constant uniform prior, why it needs no decision about wrong-class targets or a label prior, and why it preserves dark knowledge. The bounded `p - q` cross-entropy gradient explained the side observation that peaked outputs on hard examples produce large gradients, so keeping outputs humble also tames gradient norms. The RL-vs-supervised difference — I want fast convergence, only late humility — forced an annealing or, more sharply, a thresholded hinge `+ beta max(0, Gamma - H)` that charges the loss only when entropy drops too low. And the whole thing is a single closed-form reduction over the logits I already have, one knob `beta`, dropping straight into the training loop's regularizer slot with no extra passes and no change to the model, optimizer, or schedule.
