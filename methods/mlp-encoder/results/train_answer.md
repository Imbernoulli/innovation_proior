I want an agent that, dropped into a task it has never seen, becomes competent in a handful of trajectories instead of learning from scratch. The tasks come from a distribution $p(T)$ — the same cheetah with a different target velocity, the same arena with a different goal — so each task is a Markov decision process that shares structure with the others but differs in its reward or dynamics, both unknown until the agent acts. The only window onto "which task am I in" is the experience the agent collects: a stream of transitions $(s, a, r, s')$. So the real object I have to build is a module that eats a small, growing pile of those tuples and turns it into something the policy can condition on to act correctly on *this* task. Two pressures squeeze the design from opposite sides. First, meta-training itself must be cheap, and the way the field does this today — RL$^2$, MAML, ProMP, MAESN — is on-policy: you adapt from data you just collected and meta-train by collecting fresh data and discarding it after one gradient step, which costs tens to hundreds of millions of environment steps on the locomotion benchmarks. Off-policy actor-critic (SAC, TD3) is one to two orders of magnitude more sample-efficient on the very same tasks because it reuses a replay buffer, so the cheap data is sitting right there. Second, exploration under uncertainty: when the agent first lands in a new task it knows almost nothing, and under sparse rewards acting greedily on a single best guess of the task fails badly — it must probe to discover where the reward is. So the adaptation mechanism cannot merely emit a point estimate; it has to represent how unsure it is and use that uncertainty to explore. A deterministic adapter, like one gradient step from a learned initialization, gives a point estimate and no handle on exploration.

The obvious first attempt is to take an off-policy algorithm, bolt a recurrent network onto the stream of transitions, condition the policy on its hidden state, and train from the buffer — recurrent DDPG in this spirit. It is hard to train and underperforms, and the reason is structural. Meta-learning lives or dies by a matching principle: the distribution of data you adapt from at test time must match the distribution you meta-trained the adaptation rule on. At test time the agent adapts from data it just collected, which is on-policy. So feeding the adaptation rule the whole replay buffer — wildly off-policy, full of stale data from old half-trained policies — creates a brutal train/test mismatch. That is the wall, and it is why everyone stayed on-policy. But stare at the conflict: the mismatch only binds the *one* component that must behave identically at train and test, the thing that turns experience into a task summary. The policy and critic have no such constraint — they just need to learn good actions and values *conditioned on* a given task summary, and for that the whole off-policy buffer is fine, even ideal. The two roles have different data appetites, so I should stop forcing one network to do both.

I propose PEARL — Probabilistic Embeddings for Actor-Critic RL — and its load-bearing piece, the probabilistic MLP context encoder. The architecture splits the system into a task-inference part that maps experience to a latent task variable $z$, and a control part (actor and critic) that conditions on $z$ and acts. Now each is fed the data it wants: the encoder receives *recently collected*, near-on-policy context so its train and test inputs agree, while the actor and critic train on the entire off-policy buffer so meta-training is cheap. From the policy's point of view $z$ is just another part of the state; the matching principle is satisfied only where it must be. That disentanglement is what dissolves the wall, and it is forced by the data-appetite conflict, not chosen for elegance.

What is $z$, concretely? Adapting to an unknown task is RL in a partially observed MDP whose hidden state is the task identity, and the principled response to a hidden variable is to maintain a *belief* over it and act on the belief — which is also exactly the uncertainty representation the second pressure demanded. If $z$ is probabilistic and I keep a posterior over it, I can do posterior sampling for exploration: sample a task hypothesis $z$, act optimally for it for a whole episode, fold the evidence back into the posterior, and the belief narrows. Because one sampled hypothesis drives an entire episode, the exploration is temporally extended — the agent can take a coherent sequence of actions to test "maybe the goal is over there" even when no single step pays off. So $z$ must be a distribution, not a point. The posterior over the task given the experience is intractable, so I do amortized variational inference: train an inference network $q_\phi(z \mid c)$ to approximate $p(z \mid c)$ and learn $\phi$ by backpropagating through sampled $z$ via the reparameterization trick. In the penalty form I actually optimize,
$$\max_\phi \; \mathbb{E}_{T} \, \mathbb{E}_{z \sim q_\phi(z \mid c^T)} \big[\, R(T, z) - \beta \cdot D_{\mathrm{KL}}\!\big( q_\phi(z \mid c^T) \,\|\, p(z) \big) \big], \qquad p(z) = \mathcal{N}(0, I).$$
The KL term earns its place for a concrete reason: $\beta \cdot D_{\mathrm{KL}}(q_\phi(z\mid c) \| p(z))$ is a variational upper bound on the mutual information $I(z; c)$, so penalizing it forces $z$ to keep only the bits of the context the downstream objective needs and throw away the rest. Without it $z$ is free to memorize idiosyncrasies of the training tasks' transitions and overfit; with it $z$ is squeezed toward a minimal sufficient statistic of the task. So the KL does double duty — it is the prior-matching term of a variational bound *and* a compression bottleneck against overfitting — with $\beta$ the knob on how hard I compress.

Now the architecture of $q_\phi$, which is the crux. Its input is the context $c$, a set of transitions $\{(s_n, a_n, r_n, s'_n)\}$, and two properties of that set are load-bearing: it is variable-sized, and it is unordered in the sense that matters. Why unordered? Because each task is a Markov MDP, so a single transition is by itself complete evidence about the task's reward and dynamics at that point; to identify the MDP it is enough to have a *collection* of transitions, and the order in which I happened to observe them carries no extra information about which MDP I am in. So $q_\phi(z \mid c)$ should be permutation-invariant: shuffle the transitions and the inferred posterior should not change. This is a property I *know* the answer must have, so I build it in rather than hope an RNN discovers it — an RNN's hidden state depends on order, which makes it learn order-invariance from data, is slow and unstable to optimize over long horizons, and caps how many transitions I can absorb. The known recipe for a permutation-invariant function of a set is to map each element through the same per-element function and then combine the results with a symmetric operation: prototypical networks take the *mean* of embedded support examples, $(1/|S|)\sum_i h(x_i)$, and Deep Sets generalizes this to $\rho(\sum_i \phi(x_i))$. So the template is clear: encode each transition independently with a shared network, then aggregate symmetrically.

But here is exactly where I cannot copy prototypical networks, and the gap is the whole game. Their aggregation produces a single deterministic embedding — the mean of point vectors. I need $q_\phi(z \mid c)$ to be a *distribution*, and one that fuses uncertainty correctly: more transitions should make me *more certain* about the task, and averaging deterministic embeddings has no notion that ten transitions pin the task down better than one. So I have each per-transition encoder output a little Gaussian factor over $z$, $\Psi_\phi(z \mid c_n) = \mathcal{N}(\mu_n, \sigma_n^2)$ — a guess at the task with a spread — and combine these factors. If each transition is an independent piece of evidence about $z$, then by the logic of combining independent evidence the joint is proportional to the *product* of the factors,
$$q_\phi(z \mid c_{1:N}) \;\propto\; \prod_{n=1}^{N} \Psi_\phi(z \mid c_n).$$
A product is symmetric, so it is permutation-invariant for free, handles any $N$, and — crucially — fuses uncertainty. Working it out per latent dimension (the factors are diagonal, so dimensions decouple), each factor's density is proportional to $\exp\!\big(-(z-\mu_n)^2/(2\sigma_n^2)\big)$, so
$$\prod_n \exp\!\Big( -\frac{(z - \mu_n)^2}{2\sigma_n^2} \Big) = \exp\!\Big( -\tfrac{1}{2} \sum_n \frac{(z - \mu_n)^2}{\sigma_n^2} \Big),$$
which is proportional to a Gaussian. Expanding the sum and keeping only the $z$-dependent terms,
$$\sum_n \frac{(z - \mu_n)^2}{\sigma_n^2} = z^2 \Big( \sum_n \tfrac{1}{\sigma_n^2} \Big) - 2z \Big( \sum_n \tfrac{\mu_n}{\sigma_n^2} \Big) + \text{const},$$
and matching the $z^2$ and $z$ coefficients against a single $\mathcal{N}(\mu_*, \sigma_*^2)$ gives the closed form
$$\frac{1}{\sigma_*^2} = \sum_n \frac{1}{\sigma_n^2}, \qquad \mu_* = \sigma_*^2 \sum_n \frac{\mu_n}{\sigma_n^2}.$$
The two formulas say exactly what I wanted: the *precisions* add, so every additional transition strictly increases precision and shrinks the posterior variance — more evidence, tighter belief, automatically — and the mean is a *precision-weighted average*, so a transition that is confident about the task pulls the combined mean toward its guess more than an uncertain one. This is Bayesian evidence accumulation falling straight out of the product of Gaussians, still permutation-invariant, still closed-form, still handling variable $N$. Before any evidence the belief resets to the unit-Gaussian prior (and the KL keeps context posteriors tied to it); once the context is nonempty, the posterior is the product above. One numerical subtlety: if any $\sigma_n^2$ hits zero its precision blows up and a single transition dominates, so I floor the per-transition variances, $\sigma_n^2 \leftarrow \max(\sigma_n^2, 10^{-7})$, before forming the product.

That fixes everything except the per-transition encoder $f_\phi$ itself, the one piece left. It takes one transition's features — the state, action, reward, and (if included) the next state concatenated into a single vector $c_n$ — and outputs the parameters of that transition's Gaussian factor, a mean vector $\mu_n$ and a variance vector $\sigma_n^2$, so its output dimension is $2 \cdot \texttt{latent\_dim}$: the first half are the means, the second half parameterize the variances. For a fixed-size input to a fixed-size output with no sequence structure to exploit — the permutation-invariance is already handled by the product aggregation downstream — the natural choice is a plain feed-forward MLP, no recurrence, memoryless, applied identically and independently to each transition. It needs enough capacity to extract task-relevant statistics but not so much that it overfits, and since the information bottleneck $\beta \cdot$KL is already pushing against overfitting from the objective side I keep it modest: three hidden layers of 200 units with ReLU. Two setup details carry their own reasons. The hidden layers use a width-aware fan-in initialization, drawing weights uniformly from $[-1/\sqrt{\text{fan\_in}}, +1/\sqrt{\text{fan\_in}}]$ so activations neither blow up nor die across three ReLU layers, with biases set to $0.1$ to keep ReLUs on the active side of zero early so gradients flow. The output layer is initialized small, uniform in $[-3\times10^{-3}, +3\times10^{-3}]$, so at init the mean output is near $0$ and the variance preactivation is near $0$ rather than making large premature commitments. To map the raw second-half outputs to a positive variance I use softplus, $\log(1 + e^x)$ — smooth, strictly positive, and $\approx 0.69$ at input $0$ — so $\sigma_n^2 = \mathrm{softplus}(\cdot)$.

The off-policy training is where the disentanglement pays off. I build on SAC, which already gives a reparameterized stochastic policy and a soft critic learned off-policy from the buffer; I simply make the policy and critic take $z$ as an extra input. The remaining question is what gradient trains the encoder. Among reconstructing the MDP, maximizing actor returns, or training through the critic, I train through the critic — the only thing the policy needs from $z$ is enough task information to act well, and "act well" is mediated entirely through the value function, so if $z$ lets the critic predict $Q(s, a, z)$ accurately across the task then $z$ has captured exactly and only the value structure control requires. Reconstruction wastes capacity on reward/dynamics details that do not affect the optimal policy, and maximizing actor returns is a noisier, more roundabout signal. So the encoder is trained from the critic's soft-Bellman gradient plus the KL,
$$L_{\text{critic}} = \mathbb{E}_{(s,a,r,s',d)\sim B,\; z \sim q_\phi(z\mid c)} \Big[ \big( Q_\theta(s, a, z) - (r_{\text{scaled}} + (1 - d)\,\gamma\,\bar V(s', \bar z)) \big)^2 \Big], \qquad L_{\text{KL}} = \beta \cdot D_{\mathrm{KL}}\!\big(q_\phi(z\mid c) \,\|\, \mathcal{N}(0, I)\big),$$
where $\bar V$ is a target network and $\bar z$ stops gradients through $z$ on the bootstrap target so the moving encoder does not chase its own bootstrap. I keep the actor and value updates conditioned on $z$ but detach $z$ there, so the encoder's gradient comes only through the critic path; the reparameterization trick, $z = \mu + \sqrt{\sigma^2}\,\xi$ with $\xi \sim \mathcal{N}(0, I)$, carries that gradient back into $\phi$. And the *context* feeding the encoder is sampled separately from the *RL batch*: the RL batch is drawn from the whole off-policy buffer (cheap, heavily reused), while the context is sampled from recently collected data so its distribution stays close to what the encoder sees at test time. That separation is the operationalized resolution of the wall. At meta-test, with no data the posterior is the prior, so I sample $z$ from it and act for an episode — a coherent random hypothesis, temporally extended exploration — then feed the new transitions through the encoder, multiply their Gaussian factors into the posterior (which tightens by the precision-adds formula), sample a fresh $z$, and repeat. As evidence accumulates the variance shrinks and behavior slides smoothly from exploratory to exploitative: the uncertainty in $q_\phi$ *is* the exploration mechanism, with no hand-engineered bonus.

Here is the encoder I would ship — the per-transition MLP that fills the one empty slot, with the product-of-Gaussians aggregation it feeds:

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
    """Original PEARL MLP context encoder: 3 x 200 ReLU, per-transition,
    output_size = 2 * latent_dim -> [ mu | pre-softplus sigma^2 ]."""

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size           # = 2 * latent_dim
        self.hidden_activation = hidden_activation

        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:               # [200, 200, 200]
            fc = nn.Linear(in_dim, h_dim)
            fanin_init(fc.weight)
            fc.bias.data.fill_(0.1)
            self.fcs.append(fc)
            in_dim = h_dim
        self.last_fc = nn.Linear(in_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        output = preactivation
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        pass                                     # stateless: nothing to reset


def product_of_gaussians(mus, sigmas_squared):
    """Aggregate per-transition Gaussian factors -> posterior N(mu, sigma^2).
    Precisions add; mean is precision-weighted; permutation-invariant."""
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def infer_posterior(encoder, context, latent_dim):
    """context: (num_tasks, num_transitions, input_size). Returns a
    reparameterized sample z plus batched posterior parameters."""
    params = encoder(context)
    params = params.view(context.size(0), -1, encoder.output_size)
    mu = params[..., :latent_dim]
    sigma_squared = F.softplus(params[..., latent_dim:])       # positive variance
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
