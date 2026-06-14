Let me start from what actually hurts. I want an agent that, dropped into a task it has never seen, gets good in a handful of trajectories instead of from scratch. The tasks come from a distribution — same cheetah, different target velocity; same arena, different goal — and they share structure, so an agent that has lived through many of them should be able to figure out a new one fast. But the only window the agent has onto "which task am I in" is the experience it collects: a stream of transitions, tuples of (state, action, reward, next-state). So the real object I have to build is a thing that eats a small, growing pile of those tuples and turns it into something the policy can condition on to act correctly on *this* task. Everything else is downstream of getting that right.

And two pressures are squeezing this from opposite sides. The first is that meta-training itself has to be cheap. The way people do this today — RL², MAML, ProMP, MAESN — is on-policy: you adapt from data you just collected, and you meta-train by collecting fresh data and throwing it away after a gradient step. On the locomotion benchmarks that runs into the tens or hundreds of millions of environment steps just to learn the adaptation rule. That's a non-starter if I ever want this on a real robot. Meanwhile, off-policy actor-critic — SAC, TD3 — is one to two orders of magnitude more sample-efficient on the very same continuous-control tasks, because it reuses a replay buffer instead of discarding experience. So the cheap data is sitting right there in a buffer. I want to meta-train off-policy.

The second pressure is exploration under uncertainty. When the agent first lands in a new task it knows almost nothing, and under sparse rewards "act greedily on a point guess of the task" fails badly: it has to actually probe to discover where the reward is. That means the adaptation mechanism can't just emit a single best guess of the task; it has to *represent how unsure it is* and use that uncertainty to explore. A deterministic adapter — one gradient step from a learned init, say — gives me a point estimate and no handle on exploration.

So let me try the obvious thing first and watch it break, because the wall it hits tells me what the method has to be. The obvious move for "off-policy, just reuse the buffer" is: take an off-policy algorithm, bolt on a recurrent network that reads the stream of transitions and outputs a hidden state, condition the policy on that hidden state, and train the whole thing from the off-policy buffer. People have tried recurrent DDPG in this spirit. It's hard to train and it underperforms, and when I think about *why*, the reason is structural, not a tuning failure. Meta-learning lives or dies by a matching principle: the distribution of data you adapt from at test time must match the distribution you meta-trained the adaptation rule on. At test time the agent adapts from data it just collected — on-policy data. So if I want the adaptation rule to behave at test time the way it did at train time, I have to feed it on-policy data at train time. But the whole point of going off-policy was to train from the *whole replay buffer*, which is wildly off-policy — full of stale data from old, half-trained policies. Feed that to the adaptation rule and there's a brutal distribution mismatch between what it sees at meta-train and what it'll see at meta-test. That's the wall: the matching principle seems to forbid exactly the off-policy reuse I came here for. This is why everyone stayed on-policy.

Stare at the conflict, because the way out is hiding in it. The mismatch is only a problem for the *one* component that has to behave identically at train and test: the thing that turns experience into a task summary, the adaptation/inference mechanism. The *policy and critic* don't have that constraint — they just need to learn good actions and values conditioned on a given task summary, and for that, off-policy data from the whole buffer is fine, even ideal. So the two roles have different data appetites. What if I stop forcing one network to do both? Split the system in two: a task-inference part that maps experience to a task representation z, and a control part (policy + critic) that conditions on z and acts. Then I can feed each the data it wants — give the inference part near-on-policy data (so its train and test inputs match), and train the policy and critic on the entire off-policy buffer (so meta-training is cheap). The task representation z becomes just another part of the state from the policy's point of view; the only thing that has to obey the matching principle is the encoder, and I can satisfy it cheaply by sampling the encoder's context from *recently collected* data rather than the whole stale buffer. Disentangling inference from control is what dissolves the wall — the architecture is forced by the data-appetite conflict, not a stylistic choice.

Now what *is* z, concretely, and how do I infer it? The honest framing: adapting to an unknown task is RL in a partially observed MDP where the hidden part of the state is the task identity. The principled response to a hidden variable is to maintain a *belief* over it — a distribution — and act on the belief. And that's also exactly the uncertainty representation my second pressure demanded. If z is a *probabilistic* latent and I keep a posterior over it, then I can do posterior sampling for exploration: sample a task hypothesis z, act optimally for it for a whole episode, see what happens, fold the evidence back into the posterior, and the belief narrows. Because one sampled hypothesis drives an entire episode, I get temporally extended exploration — the agent can take a sequence of actions to test "maybe the goal is over there" even when no single step pays off immediately. So z wants to be a distribution, not a point. That settles it: I need a posterior p(z | context), and a way to sample from it and condition the policy on the sample.

The posterior over the task given the experience is intractable, so I'll do what works for intractable posteriors: amortized variational inference. Train an inference network q_φ(z | c) to approximate p(z | c), and learn its parameters by optimizing through sampled z via the reparameterization trick. In the penalty form I actually optimize, the objective is an expected downstream return under z drawn from q, traded off against a KL term that ties q to a prior p(z) which I'll take to be a unit Gaussian:

  max_φ E_{T} E_{z ∼ q_φ(z | c^T)} [ R(T, z) − β · D_KL( q_φ(z | c^T) ‖ p(z) ) ],

where R is the downstream objective (I'll pin it down shortly). Equivalently, in the code path that minimizes losses, I add `β·D_KL(q_φ(z|c) || p(z))` to the critic-side encoder loss.

Let me make sure I understand that KL term, because I want it for a concrete reason and not by reflex. Read it as an information bottleneck: β·KL(q(z|c) ‖ p(z)) is a variational upper bound on the mutual information I(z; c) between the latent and the context. Penalizing it forces z to keep only the bits of the context the downstream objective actually needs and to throw away the rest. In meta-learning that's precisely the regularizer I want — without it, z is free to memorize idiosyncrasies of the specific transitions in the training tasks and overfit; with it, z is squeezed toward "minimal sufficient statistic of the task." So the KL term is doing double duty: it's the prior-matching term of a variational bound *and* a compression bottleneck against overfitting. β is the knob on how hard I compress.

Now the architecture of q_φ — this is where the encoder lives, and it's the crux. The input to q_φ is the context c, which is a *set* of transitions {(s_n, a_n, r_n, s'_n)}, and two properties of that set are load-bearing. It's *variable-sized* — the agent has collected anywhere from a handful to many transitions — and it's *unordered* in the sense that matters. Why unordered? Because each task is a Markov MDP: a single transition (s, a, r, s') is, by itself, complete evidence about the task's reward and dynamics at that point. To identify the MDP — to nail down the reward function and the transition function — it's enough to have a *collection* of transitions; the order in which I happened to observe them carries no extra information about which MDP I'm in. So the right inference network for q_φ(z | c) should be *permutation-invariant*: shuffle the transitions and the inferred posterior should not change.

This is exactly the constraint that should drive the design, so let me take it seriously rather than reach for an RNN. An RNN reads the transitions in order and its hidden state depends on that order — it imposes a sequence structure the problem doesn't have. That's not just wasteful; it's the source of the trouble. The order-dependence makes the RNN have to *learn* to be (approximately) order-invariant from data, it's slow and unstable to optimize over long horizons, and it caps how many transitions I can absorb. If permutation-invariance is a property I *know* the answer must have, I should *build it in*, not hope the network discovers it.

How do you build a permutation-invariant function of a set? The known recipe is to map each element through the same per-element function and then combine the results with a *symmetric* operation — a sum, a mean, a product — something that doesn't care about order. Prototypical networks do exactly this for few-shot classification: the class prototype is the *mean* of the embedded support examples, (1/|S|) Σ_i h(x_i) — embed each element, average. Deep Sets makes it general: any permutation-invariant function of a set can be written ρ(Σ_i φ(x_i)) — a per-element map, summed, then transformed. So the template is clear: encode each transition c_n independently with a shared network, then aggregate with a symmetric reduction.

But here's where I can't just copy prototypical networks, and the gap is the whole game. Their aggregation produces a single *deterministic* embedding — the mean of point vectors. I need q_φ(z | c) to be a *distribution*, and not just any distribution: I need the aggregation to *fuse uncertainty* correctly. More transitions should make me *more certain* about the task, and the way I combine per-transition information has to reflect that. Averaging deterministic embeddings doesn't do that — averaging ten confident guesses and averaging ten unsure guesses both just give you a mean, with no notion that ten transitions pin the task down better than one. So I need a symmetric aggregation that lives in distribution space and tightens as evidence accumulates.

Let me set it up as combining per-transition *opinions*, each opinion a distribution over z. Have the per-transition encoder output, for each transition c_n, a little Gaussian factor over z — a guess at the task with a spread, Ψ_n(z) = N(μ_n, σ_n²) — and combine these factors into the posterior. What symmetric operation combines distributions into a sharper joint belief? If each transition is treated as an independent piece of evidence about z, then by the logic of combining independent evidence the joint is *proportional to the product of the factors*:

  q_φ(z | c_{1:N}) ∝ ∏_{n=1}^{N} Ψ_φ(z | c_n).

A product is symmetric — multiplication commutes, so it's permutation-invariant for free. It handles any N. And — this is the part averaging couldn't do — let me actually compute what a product of diagonal Gaussian factors is proportional to and check that it sharpens with evidence. Take one latent dimension; the factors are diagonal so dimensions decouple. Each factor's density is proportional to exp(−(z − μ_n)² / (2σ_n²)). Multiply them:

  ∏_n exp( −(z − μ_n)² / (2σ_n²) ) = exp( −½ Σ_n (z − μ_n)² / σ_n² ).

The product is proportional to a Gaussian, so the normalized result is N(μ*, σ*²) for some μ*, σ*² I can read off by matching the exponent's dependence on z. Expand the sum, keeping only the z-dependent pieces:

  Σ_n (z − μ_n)² / σ_n² = z² · ( Σ_n 1/σ_n² ) − 2z · ( Σ_n μ_n/σ_n² ) + (const in z).

A single Gaussian N(μ*, σ*²) has exponent −½ ( z²/σ*² − 2z μ*/σ*² + … ). Match the z² coefficient and the z coefficient:

  1/σ*² = Σ_n 1/σ_n²,        μ*/σ*² = Σ_n μ_n/σ_n²,

so

  σ*² = 1 / Σ_n (1/σ_n²),     μ* = σ*² · Σ_n (μ_n / σ_n²).

Look at what those two formulas say. The *precisions* (the 1/σ²) add up: the combined precision is the sum of the per-transition precisions, so every additional transition strictly increases the precision and therefore *shrinks* the posterior variance σ*² — more evidence, tighter belief, automatically. And the mean is a *precision-weighted average* of the per-transition means: a transition that's confident about the task (small σ_n², large precision) pulls the combined mean toward its guess more than an uncertain one. This is precisely the uncertainty-fusing behavior averaging couldn't give me — it's Bayesian evidence accumulation falling straight out of the product of Gaussians. And it's still permutation-invariant, still closed-form, still handles variable N. The product is the aggregator; the per-transition Gaussian factor is what the encoder must produce. The empty-context boundary condition is separate: before any evidence, I reset the belief to the unit Gaussian prior, and during training the KL penalty keeps context posteriors tied to that prior. Once I have a nonempty context, the closed form above is the product over the encoder's observed-transition factors.

There's a subtlety I should pin before it bites me numerically: if any σ_n² is allowed to hit zero, its precision 1/σ_n² blows up and a single transition dominates everything — and division by zero looms. So I'll floor the per-transition variances at a small positive value before forming the product. A clamp like σ_n² ← max(σ_n², 1e-7) keeps the precisions finite and the aggregation well-behaved no matter what the network spits out early in training.

Now I can finally say what the per-transition encoder f_φ has to be, and it's the only piece left unspecified. It takes one transition's features — concatenate the state, action, reward, and (if I include it) the next state into a single input vector c_n — and outputs the parameters of that transition's Gaussian factor: a mean vector μ_n and a variance vector σ_n², each of dimension equal to the latent z. So its output dimension is twice the latent dimension: the first half are the means, the second half parameterize the variances. For a fixed-size input vector to a fixed-size output vector, with no sequence structure to exploit (the permutation-invariance is already handled by the product aggregation downstream), the natural choice is a plain feed-forward MLP. No recurrence — that was the whole point of choosing a permutation-invariant aggregation; the encoder itself is memoryless and applies identically and independently to each transition. It needs enough capacity to extract the task-relevant statistics from a transition but not so much that it overfits — and the information bottleneck β·KL is already pushing against overfitting from the objective side, so I can keep the encoder a modest size and lean on the bottleneck. Three hidden layers of a couple hundred units with ReLU nonlinearities is the standard capacity for this kind of continuous-control function approximator; I'll use 200-200-200, ReLU.

Two details of how I set up that MLP matter and I want their reasons, not just the numbers. First, initialization of the hidden layers. With ReLU units I want a width-aware small uniform initialization so activations do not immediately blow up or die as signal passes through three layers. The helper I will use draws a 2-D weight tensor uniformly from [−1/√size(0), +1/√size(0)], and I will keep that exact bound. I'll also give the biases a small positive value, 0.1, a mild nudge to keep ReLUs on the active side of zero early in training so gradients flow. Second, the output layer should not begin by making large commitments about either the mean or the variance logits. So I initialize the final layer's weights and biases to very small values, uniform in [−3e-3, +3e-3]. Then at init the mean output is near 0 and the variance preactivation is near 0. Which brings up: how do I turn the raw second-half outputs into a *positive* variance? A variance must be positive, and I'd like the map smooth and to give a reasonable spread at a near-zero input. Softplus, log(1 + e^x), does both — it's smooth, strictly positive, and at input 0 gives log 2 ≈ 0.69. So σ_n² = softplus(second half of the encoder output). The prior itself comes from the reset state with mean 0 and variance 1; the small output initialization just keeps learned factors from starting with huge means or degenerate variances.

Now the off-policy training, where the disentanglement pays off concretely. I build on SAC. SAC already gives me a reparameterized stochastic policy and a soft critic learned off-policy from the buffer; I just make the policy and critic take z as an extra input alongside the state. The question is: what gradient trains the encoder? I have a few candidates. I could train it to reconstruct the MDP — predict next states and rewards from z (a generative objective). I could train it to maximize the actor's returns. Or I could train it through the *critic*: make z be whatever makes the state-action value Q(s, a, z) predictable, i.e. let the encoder receive gradients from the Bellman error of the critic. The reasoning that picks the third: the only thing the policy actually needs from z is enough task information to act well, and "act well" is mediated entirely through the value function — if z lets the critic predict Q accurately across the task, then z has captured the task's value structure, which is exactly and only what control requires. Reconstruction wastes capacity modeling reward/dynamics details that don't affect the optimal policy, and maximizing actor returns is a noisier, more roundabout signal. So I train the encoder from the critic's Bellman gradient, plus the β·KL bottleneck term. The critic loss is the soft Bellman residual with z drawn from the encoder's posterior:

  L_critic = E_{(s,a,r,s',d) ∼ B, z ∼ q_φ(z|c)} [ ( Q_θ(s, a, z) − ( r_scaled + (1 − d) γ V̄(s', z̄) ) )² ],

where V̄ is a target network and z̄ means I stop gradients through z on the bootstrap target (you don't want the moving encoder to chase its own bootstrap). I keep the actor and value updates conditioned on z but *detach* z there, so the encoder's gradient comes through the critic path only — that's the clean signal I argued for. The reparameterization trick lets the critic's gradient flow back through the sampled z into φ. And critically, I sample the *context* that feeds the encoder separately from the *RL batch* that feeds actor and critic: the RL batch is drawn from the whole off-policy buffer (cheap, lots of reuse), while the context is sampled from *recently collected* data so its distribution stays close to the on-policy data the encoder will see at test time. That separation is the resolution of the original wall, now operationalized: one component (control) eats the stale buffer, the other (inference) eats fresh data, and the matching principle is satisfied only where it has to be.

Let me also nail the test-time loop, since it's the payoff of making z probabilistic. New task, no data: the posterior is the prior, so I sample z from the prior and act for an episode — that's a coherent random hypothesis, temporally extended exploration. Collect that trajectory, feed its transitions through the encoder, multiply their Gaussian factors into the posterior (which, by the precision-adds formula, tightens), sample a fresh z from the *updated* posterior, act another episode. As evidence accumulates the variance shrinks and z concentrates on the true task, so the agent's behavior slides smoothly from exploratory (broad posterior, diverse hypotheses) to exploitative (tight posterior, near-optimal for the identified task). That's posterior sampling, meta-learned: no hand-engineered exploration bonus, the uncertainty in q_φ *is* the exploration mechanism.

So let me write the encoder I'd actually ship — the per-transition MLP that fills the one empty slot, with the product-of-Gaussians aggregation it feeds. The encoder is a stack of fan-in-initialized ReLU layers ending in a small-initialized linear layer to 2·latent_dim, stateless, applied independently to every transition:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = int(np.prod(size[1:]))
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MlpEncoder(nn.Module):
    """Stateless MLP context encoder. Maps one transition's features
    (s, a, r [, s']) -> output_size = 2 * latent_dim parameters:
    [ mu_n | pre-softplus sigma_n^2 ] for that transition's Gaussian factor.
    Stateless and shared across transitions; permutation-invariance is
    handled by the product-of-Gaussians aggregation downstream."""

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size          # = 2 * latent_dim
        self.hidden_activation = hidden_activation

        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:              # e.g. [200, 200, 200]
            fc = nn.Linear(in_dim, h_dim)
            fanin_init(fc.weight)               # ReLU-friendly fan-in init
            fc.bias.data.fill_(0.1)             # keep ReLUs active early
            self.fcs.append(fc)
            in_dim = h_dim
        self.last_fc = nn.Linear(in_dim, output_size)
        # small init: keep initial factor outputs near zero
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, transition_features, return_preactivations=False):
        h = transition_features
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        output = preactivation                  # [ mu | sigma^2-preactivation ]
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        pass                                    # memoryless: nothing to reset


def product_of_gaussians(mus, sigmas_squared):
    """Combine per-transition Gaussian factors N(mu_n, sigma_n^2) into the
    posterior N(mu, sigma^2). Precisions add (belief sharpens with evidence);
    mean is the precision-weighted average. Permutation-invariant by construction."""
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)   # floor variances
    sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def infer_posterior(encoder, context, latent_dim):
    """context: (num_tasks, num_transitions, input_size)."""
    params = encoder(context)
    params = params.view(context.size(0), -1, encoder.output_size)
    mu = params[..., :latent_dim]
    sigma_squared = F.softplus(params[..., latent_dim:])     # positive variance
    z_params = [
        product_of_gaussians(m, s)
        for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
    ]
    z_means = torch.stack([p[0] for p in z_params])
    z_vars = torch.stack([p[1] for p in z_params])
    posteriors = [
        torch.distributions.Normal(m, torch.sqrt(s))
        for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))
    ]
    z = torch.stack([d.rsample() for d in posteriors])
    return z, z_means, z_vars
```

Let me retrace the causal chain so I'm sure each piece earned its place. I wanted fast, uncertainty-aware adaptation with cheap meta-training, and the cheap data only exists in an off-policy buffer — but the meta-learning matching principle seemed to forbid using it, since at test time the agent adapts on-policy. The mismatch only binds the component that must behave the same at train and test, the inference mechanism; the policy and critic don't care, so I split the system into task inference and control and fed each the data it wants, getting around the wall. Adaptation under a hidden task is belief inference in a POMDP, and a *probabilistic* latent z lets me both represent uncertainty and explore by posterior sampling, so z is a distribution learned by amortized variational inference with a β·KL penalty that doubles as an information bottleneck against overfitting. The context is a variable-sized, unordered set of Markov transitions, so the inference network must be permutation-invariant, which the known set-encoding recipe (encode each element, combine symmetrically) supplies — but I needed the symmetric combination to fuse *uncertainty*, which a mean of deterministic embeddings can't, so I made each transition emit a Gaussian factor and aggregated by *product*: computing it out, the precisions add (belief sharpens with evidence) and the mean is precision-weighted, exactly Bayesian fusion, closed-form and order-free. The per-transition factor is produced by a plain stateless MLP — three 200-unit ReLU layers, fan-in init with the codebase's bound, small final-layer init to avoid large initial factor outputs, softplus to keep variance positive — applied independently to each transition. The encoder is trained off-policy by the critic's Bellman gradient (plus the KL loss), because all the policy needs from z is the task's value structure; z is detached on the bootstrap target and into the actor/value, and the context is sampled from recent data while the RL batch comes from the whole buffer. At test time, sample z from the reset prior, act, multiply the new transitions' factors into the posterior to tighten it, resample — exploration that anneals itself from broad hypotheses to the identified task as the belief narrows.
