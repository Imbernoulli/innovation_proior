Let me start from what actually hurts. I drop an agent into an environment it has never seen, drawn from a family of related environments — same state and action space, but the reward and the dynamics differ from task to task, and the agent doesn't know which task it's in. It can interact. But the thing I'm graded on is the return it accumulates *while* it's figuring the environment out, from the very first steps, not after some long separate adaptation. So every action is a gamble: do I poke around to learn what kind of environment this is, or do I cash in on what I already suspect? Get that trade-off wrong and I either flail forever or commit to a wrong guess. I want the agent to explore exactly as much as it's worth, given how much time it has left, and not a step more.

The clean way to say "exactly as much as it's worth" is to act optimally with respect to my own uncertainty about which environment I'm in. If I genuinely don't know whether the goal is to the left or the right, the right thing is to gather the information that resolves it — but only if resolving it lets me earn more before the horizon than the gathering costs. That's a policy that conditions its action not just on the state, but on the *belief* about the task. And there's a textbook object for exactly this: the Bayes-Adaptive MDP. Put a prior `b_0 = p(R, T)` over the unknown reward and transition functions. As experience `tau_{:t} = (s_0, a_0, r_1, s_1, ..., s_t)` comes in, carry a posterior `b_t = p(R, T | tau_{:t})`. Glue this belief onto the environment state to make a hyper-state `s^+_t = (s_t, b_t)`. The hyper-state transitions are forced on me: the environment part moves under the current posterior over dynamics, `E_{b_t}[T(s_{t+1}|s_t,a_t)]`, and the belief part updates deterministically by Bayes' rule, `b_{t+1} = p(R,T|tau_{:t+1})` — that's the `delta(...)` in the transition. The reward on hyper-states is the expected reward under the updated posterior, `E_{b_{t+1}}[R]`. Maximise `J^+(pi) = E[sum_t gamma^t R^+]` over a horizon `H^+`, and the maximiser is the Bayes-optimal policy. By construction it takes an information-seeking action only when reducing uncertainty raises expected return within the horizon. And `H^+` is its own knob: I might want the agent Bayes-optimal across the first `N` episodes, `H^+ = N x H`, because whether a costly probe is worth it depends entirely on how many episodes I'll get to exploit the answer.

So the *specification* is perfect. Why doesn't everyone just do this? Because solving a BAMDP is hopeless past toy size, and it's hopeless in three distinct places, which I should pin down separately because the fixes will be different. First, I usually don't even know how to *parameterise* the true reward and transition functions — what family `p(R,T)` lives in. Second, the belief update — computing the posterior `p(R,T|tau_{:t})` — is itself intractable for any rich model. Third, even handed the exact posterior, planning in belief space, i.e. optimising `J^+` over hyper-states, is intractable. Three walls. Tabular Bayesian-RL planners (sample-based tree search over the belief, hand-specified prior and Bayes update) get through none of them at scale: they live in small discrete state/action spaces or a small discrete set of tasks, they need me to hand-write the prior and the belief update, and the planning is expensive. Dead end for deep RL and continuous control.

What's the cheap escape everyone reaches for? Posterior sampling — Thompson's old trick, lifted to RL by Strens and by Osband. Don't plan in belief space; instead, at the start of each episode, *sample* one hypothesis MDP from the posterior, compute the policy that's optimal for that single sampled MDP, and follow it until you resample. That dodges the third wall entirely: planning is now on an ordinary MDP, not a BAMDP. And it does explore — by randomising over the uncertainty. But picture it in the gridworld where the goal is somewhere in a region. Posterior sampling picks a candidate goal cell, walks straight to it, finds nothing, updates, picks *another* candidate, walks there, and so on. It commits to one hypothesis at a time and chases it. It revisits cells, it doesn't sweep the possibilities in the systematic order that minimises expected steps-to-goal, and so its return-during-learning is well below what an uncertainty-aware searcher would get. A genuinely Bayes-optimal searcher would walk a path that rules out as many cells as possible per step, never doubling back, because it reasons about the *whole* posterior, not one draw from it. So posterior sampling is the tractable-but-inefficient corner, and the prize I'm after is: get the *efficiency* of Bayes-optimal exploration while keeping the *tractability* of posterior sampling. And posterior sampling still leaves the second wall standing — it needs the posterior, which I said is intractable.

Let me line up the meta-learning tools too, because I'm not solving one fixed MDP — I have a *distribution* `p(M)` I can sample from at training time, and that changes everything. The whole point of meta-learning is to use those related training tasks to learn how to adapt fast on a new one. What's on the table? RL² — "learning to reinforcement learn." Make the policy a recurrent net; feed it, at each step, the observation *plus the previous action, previous reward, and a done flag*; keep the hidden state running across episodes within a trial on a fixed MDP; and train, by ordinary RL, to maximise return over the whole trial. The within-task learning then happens entirely inside the recurrent dynamics — slow outer-loop RL distills a fast inner learner into the RNN's weights. And there's a beautiful argument, from the memory-based-meta-learning literature, for why this is even principled: when you train a recurrent agent across a task family with its own past data as input, the hidden state is pressured to track whatever statistics it needs to act well — it ends up amortising something like a Bayes filter, carrying sufficient statistics of the posterior in the recurrence. So a recurrent net is, in principle, capable of approximately Bayes-optimal behaviour, and recurrence is the right substrate for *online* inference: it naturally folds each new transition into a running state, one update per step, exactly the shape of a belief update.

But RL² as it stands has a hole I keep circling back to: there is no explicit representation of the task, and crucially no explicit representation of *uncertainty*. The hidden state is an opaque vector trained by one signal only — the policy gradient. Nothing pressures it to encode "which task is this" beyond whatever the reward happens to demand, and nothing pressures it to carry a calibrated "how sure am I." It's a black box that *might* learn to filter, with no inductive bias toward doing so. And empirically that opacity bites: a big recurrent hidden state gets unstable across episode resets, because the state-space jumps and the hidden vector wasn't built to be a stable task descriptor. The other meta-RL families are worse for my purpose. MAML and its descendants meta-learn an initialisation so a few gradient steps adapt — but they collect the exploration rollouts *under the un-adapted policy, before* adapting, then exploit *after*. Exploration and exploitation are split into separate phases by construction, so the exploration is never optimised for online return; that's the opposite of what Bayes-optimality wants, where the two are interleaved and traded off continuously. PEARL encodes the context into a probabilistic latent and trains off-policy with SAC, which is sample-efficient — but its encoder is *permutation-invariant*, a product of per-transition Gaussians, so it treats the collected transitions as an unordered set, assuming they're independent and throwing away the temporal structure of exploration; and at action time it *samples* a latent and acts greedily, which is just posterior sampling again, inefficient online. And the one method that does condition the policy on a posterior over the MDP, Humplik's task-inference, trains that inference with *privileged* task labels — true task descriptions I usually won't have.

So the gap is fourfold. I want the policy to condition on task *uncertainty* (that's what makes exploration efficient and Bayes-optimal-ish), I want the inference to be *online* and to respect the sequential structure of experience (that's recurrence), I want it *tractable and scalable* to deep nets and continuous control (no belief-space planning, no hand-built prior), and I want it *unsupervised* (no privileged task labels). No single method has all four. Let me try to build one.

Start with the first wall: I don't know how to parameterise the true `R, T`. The trick is not to parameterise them in their native, million-parameter form at all. The tasks share structure; what differs between `M_i` and `M_j` is some low-dimensional thing — a goal position, a target velocity. So introduce a learned, low-dimensional stochastic latent `m` that *stands in* for the task, and write `R_i(r|s,a,s') ~= R(r|s,a,s'; m_i)` and `T_i(s'|s,a) ~= T(s'|s,a; m_i)`, where `R` and `T` are now single networks *shared across all tasks*, modulated by `m`. This is a big move and I want to be sure it's the right one. It collapses the belief over `(R,T)` — over millions of parameters — into a belief over a small vector `m`. Far fewer things to infer over, and the shared `R, T` get to learn from *every* task's data instead of being re-estimated per task. The whole problem reduces to: maintain a posterior over `m`, and condition the policy on it. The first wall is gone — I never parameterise the true `R,T`; I parameterise a latent and a shared decoder.

Now the second wall: the posterior over `m`. I can't compute `p(m | tau_{:t})` exactly — there's no access to the true MDP, and marginalising over tasks is infeasible. This is exactly the situation amortised variational inference was built for. Learn a recognition network `q_phi(m | tau_{:t})` that maps the experience-so-far to (the parameters of) an approximate posterior, and learn a generative model of the trajectory `p_theta(tau | m)`. Train them together to make the model assign high likelihood to the data. The model-learning objective is to maximise, over the trajectory distribution `rho` that my own policy induces,

  E_{rho(M, tau)}[ log p_theta(tau) ].

That marginal likelihood is intractable — it integrates `m` out. But the standard variational move turns it into something I can optimise. Let me actually do the derivation rather than quote the bound, because every term is going to matter. Take the log-marginal and slip in the recognition density as a ratio that equals one:

  log p_theta(tau) = log ∫ p_theta(tau, m) dm = log ∫ p_theta(tau, m) [ q_phi(m|tau_{:t}) / q_phi(m|tau_{:t}) ] dm = log E_{q_phi(m|tau_{:t})}[ p_theta(tau, m) / q_phi(m|tau_{:t}) ].

Now `log` is concave, so by Jensen I can pull it inside the expectation and only lose:

  log E_q[ p_theta(tau,m)/q_phi ] >= E_{q_phi(m|tau_{:t})}[ log ( p_theta(tau, m) / q_phi(m|tau_{:t}) ) ].

Split the log of the ratio. Factor the joint as `p_theta(tau, m) = p_theta(tau | m) p_theta(m)`:

  = E_q[ log p_theta(tau|m) + log p_theta(m) - log q_phi(m|tau_{:t}) ]
  = E_{q_phi(m|tau_{:t})}[ log p_theta(tau | m) ] - KL( q_phi(m|tau_{:t}) || p_theta(m) ).

There's my evidence lower bound. The first term is a reconstruction term — under a sampled `m` from the posterior, how well does the decoder explain the trajectory. The second is a KL pulling the posterior toward a prior `p_theta(m)`. I'll call it `ELBO_t`, and I can estimate it by Monte Carlo, sampling `m ~ q_phi` with the reparameterisation trick `m = mu + sigma . eps`, `eps ~ N(0,I)`, so the gradient flows through `mu, sigma`. The second wall is down: the posterior is whatever `q_phi` outputs, computed in a single forward pass — fast enough to run *online, at every timestep*, which is what an agent acting under uncertainty needs.

And the third wall — planning in belief space — I'm going to dissolve rather than climb. Instead of computing the belief and then planning against it, I'll *condition the policy directly on the posterior* and train the policy end-to-end by RL. No explicit planning at test time at all; the policy has learned, during meta-training, to act well as a function of its current belief. So the architecture is: a posterior over `m`, fed to a policy, plus a decoder that supplies the learning signal for the posterior.

But wait — if I literally apply a vanilla VAE here I'll get something underwhelming, and I need to figure out what to actually reconstruct, because the obvious choice is subtly wrong. A conventional VAE encodes its input and reconstructs *that same input*. If I do that — encode `tau_{:t}`, decode `tau_{:t}` — the latent `m` only needs to compress the history I've already seen. That's not what I want. I want `m` to capture *the task*, because the task is what lets the agent predict what it hasn't seen yet and explore intelligently. The encoder can only ever see the *past* `tau_{:t}` — that's all the information available at decision time `t` — so it forms the posterior `q(m | tau_{:t})` from the prefix. But there's no reason the *decoder* has to stop at the prefix. At training time I have the whole rollout, so I can ask the decoder: given an `m` inferred from only the first `t` transitions, predict the states and rewards of the *entire* trajectory `tau_{:H^+}`, future included. That's impossible unless `m` has captured the underlying task — the goal, the target velocity — rather than just memorising the seen prefix. So decoding the future, while encoding only the past, forces `m` to be a task descriptor and teaches the agent to reason about unseen states from limited experience. It's a Jaderberg-style auxiliary loss: reconstruction as an unsupervised signal that shapes the representation, layered on top of RL.

Let me write that reconstruction term out, because its factorisation tells me what networks I actually need. Conditioned on `m` and the actions, the trajectory likelihood factorises over time into an initial-state term, per-step transition terms, and per-step reward terms:

  log p(tau_{:H^+} | m, a_{:H^+-1}) = log p(s_0 | m) + sum_{i=0}^{H^+-1} [ log p(s_{i+1} | s_i, a_i, m) + log p(r_{i+1} | s_i, a_i, s_{i+1}, m) ].

So the decoder is two heads sharing `m`: a transition decoder `T'(s_{i+1} | s_i, a_i; m)` and a reward decoder `R'(r_{i+1} | s_i, a_i, s_{i+1}; m)` (the initial-state term folds into `T'`). These are exactly the generalised, latent-conditioned reward and transition functions from the latent reformulation — the shared `R, T` modulated by `m`. Good, it all hangs together: the decoder *is* the environment model, and the latent it conditions on *is* the task belief.

And now the question of what `t` should be. In the bound I wrote one `ELBO_t` for a particular context length `t`. But the agent acts at *every* timestep, and at each one it has a different amount of experience — it should have a sharp-enough posterior after one step, a sharper one after ten. If I only train the ELBO at the final step, the encoder never learns to give a good posterior *early*, which is precisely when exploration decisions matter most. So I should train the ELBO at *all* context lengths and sum them. That makes the encoder learn *online* inference: produce a useful posterior `q(m|tau_{:t})` for every `t`, so the agent's uncertainty contracts smoothly as data arrives. The model objective becomes `sum_{t=0}^{H^+} ELBO_t`, and if `H^+` is large I can subsample a handful of `t`'s per update for efficiency. Combine with the RL objective `J` for the policy, weighting the supervised model term against the RL term with a coefficient `lambda` (necessary because the encoder parameters `phi` are shared between the model and the policy):

  L(phi, theta, psi) = E_{p(M)}[ J(psi, phi) + lambda * sum_{t=0}^{H^+} ELBO_t(phi, theta) ].

Now the prior in each `ELBO_t`'s KL. The lazy choice is a fixed `N(0,I)` for every `t`. But think about what the KL is doing: `KL(q(m|tau_{:t}) || p(m))` regularises the posterior toward the prior. If the prior is the *same* fixed Gaussian at every step, then at every step I'm pulling the belief back toward total ignorance — which fights the contraction I want. What I actually want is a Bayesian filter: my belief at step `t` should be regularised toward my belief at step `t-1`, not toward the origin. So set the prior in `ELBO_t` to the *previous posterior*, `p(m) := q_phi(m | tau_{:t-1})`, while the empty-history posterior is itself regularised against `N(0,I)`. Now the KL chain penalises *changing my mind without evidence* — it lets the posterior sharpen as data justifies it, but discourages it from jolting around step to step, and it still anchors the first belief to a unit Gaussian. Concretely, with diagonal Gaussians `q(m|tau_{:t}) = N(mu_t, S_t)` and previous `N(mu_{t-1}, S_{t-1})`, the per-step KL is the full Gaussian-to-Gaussian formula, not the simple-to-`N(0,I)` one:

  KL( N(mu_t, S_t) || N(mu_{t-1}, S_{t-1}) ) = 0.5 [ log|S_{t-1}|/|S_t| - K + tr(S_{t-1}^{-1} S_t) + (mu_{t-1}-mu_t)^T S_{t-1}^{-1} (mu_{t-1}-mu_t) ],

with `K` the latent dimension; for diagonal covariances every trace, log-det, and quadratic form is just a sum over coordinates. The first term is the same formula with `mu_{-1}=0` and `S_{-1}=I`, i.e. `KL(q(m|tau_:0) || N(0,I))`. If I instead want every posterior pulled to the fixed unit Gaussian, the formula becomes the simpler `KL(q || N(0,I)) = -0.5 sum (1 + log sigma^2 - mu^2 - sigma^2)`.

Now the encoder itself. The posterior must be computed *online*, folding in one transition at a time, and it must respect the *order* of experience — what I should do next depends on the whole sequence so far, not on an unordered bag of transitions. That rules out the permutation-invariant set encoder; the right primitive is recurrent. So: embed each `(a, s, r)` with small feature extractors, concatenate, run a GRU whose hidden state carries the running summary, and read out `mu` and `log sigma^2` of the Gaussian posterior from the hidden state. One GRU step per environment step = one belief update per environment step, which is exactly the online structure I wanted; and because it's recurrent, feeding the whole trajectory through in one pass gives me the posterior at *every* prefix length `t` at once — precisely the `q(m|tau_{:t})` for all `t` that the summed ELBO needs. Across ordinary episode boundaries inside the same task I keep that hidden state, because the BAMDP horizon is allowed to span multiple rollouts; I zero it only when the BAMDP/task is done and a fresh task begins. And notice what falls out: if I take this whole construction and *delete* the decoder and the entire ELBO objective, I'm left with a recurrent policy fed `(s, a, r)` trained by RL — that's RL². So RL² is the special case of my method with no inference machinery. The differences that remain are exactly the inductive biases I argued I needed: a *stochastic* latent (the Gaussian, which can represent uncertainty — an opaque hidden vector cannot say "I'm not sure yet," and being able to say so is what drives early exploration), and the decoder reconstructing past *and future* transitions and rewards (the auxiliary loss that forces the task into the latent and lets the agent infer about unseen states). The unification tells me I haven't invented a fourth competitor; I've found the principled object that RL² was a crude approximation to.

How does the policy condition on the belief? This is where I have to resist the obvious-but-wrong choice once more. The tempting thing is to *sample* `m ~ q(m|tau_{:t})` and feed the sample to the policy. But that's posterior sampling — act as if one draw were the truth, and you get exactly the inefficient committing-to-a-hypothesis behaviour I'm trying to beat. Instead, condition the policy on the *posterior itself* — both the mean and the variance: `pi(a_t | s_t, q(m|tau_{:t}))`, implemented by concatenating `(mu_t, log sigma^2_t)` with the state. Now the policy can *see* its own uncertainty and act on it: when the variance is high it can choose to probe; when it's low it can exploit. That's the BAMDP hyper-state `(s_t, b_t)` made concrete, with the learned latent belief playing the role of `b_t`. Feeding the distribution rather than a sample is the difference between Bayes-optimal-style reasoning and posterior sampling — it's the same distinction that separated efficient sweeping from random-goal-chasing back in the gridworld, now baked into what the policy gets to see.

One more design question, and it's a practical one that turns out to matter a lot. The encoder parameters `phi` sit in two objectives — the ELBO and, through the policy's input, the RL loss. Should the RL gradient flow back through the encoder? My first instinct is yes, end-to-end. But two things push back. First, expense: I train the policy with an on-policy method that does several minibatch passes per update; if the RL loss backprops through the recurrent encoder, I have to recompute the embeddings on every one of those passes — many extra forward/backward passes through a GRU, which is the slowest part. Second, interference: the RL loss and the VAE loss pull `phi` in different directions, and mixing them forces me to tune `lambda` carefully to keep them from fighting. If I *stop* the RL gradient at the encoder — train the VAE (encoder + decoder) with one optimiser and buffer, the policy with a separate optimiser, learning rate, and buffer (recent on-policy data for the policy; a larger separate trajectory buffer for the VAE) — then the encoder is shaped *only* by the ELBO, the embeddings can be reused across the policy's minibatch passes, and there's no loss to trade off. The decoder's reconstruction is rich enough on its own to force `m` to encode the task, so the RL signal through the encoder turns out unnecessary in practice. So: detach the latent before it enters the policy. (And I can simplify further per domain — on the continuous-control families a *reward*-only decoder, dropping the transition head, works well even where the dynamics change, since the reward usually carries the task identity and it's cheaper.)

At test time the picture is clean and that's the payoff of dissolving the planning wall. Roll the policy out on a fresh held-out task with forward passes only: feed each new `(s, a, r)` into the encoder to update the posterior `(mu, log sigma^2)`, hand state-plus-posterior to the policy, act, repeat. The decoder is not used. No gradient steps, no inner-loop adaptation, no privileged task label — the policy has *already* learned, during meta-training, to act approximately Bayes-optimally as a function of its belief, so adaptation is just the belief sharpening as experience accrues, in a single forward pass per step.

Let me now turn this into the code I'd actually run, filling the empty slots of the harness — the task-representation module (a recurrent VAE encoder over `(s,a,r)`), the decoders that train it, and the policy that conditions on the posterior. The encoder reads out a per-step Gaussian and, run over a whole trajectory with the prior prepended, returns a belief at every prefix length:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    """q_phi(m | tau_:t): embed each (a, s, r), run a GRU online, read out the
    Gaussian posterior (mu, logvar) over the task latent m from the hidden state.
    One GRU step = one belief update; a full-trajectory pass yields q(m|tau_:t)
    for every prefix t at once (with the N(0,I) prior prepended)."""

    def __init__(self, latent_dim=5, hidden_size=128,
                 action_dim=2, state_dim=2,
                 a_embed=16, s_embed=32, r_embed=16):
        super().__init__()
        self.latent_dim, self.hidden_size = latent_dim, hidden_size
        self.action_encoder = nn.Linear(action_dim, a_embed)   # small feature extractors
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.reward_encoder = nn.Linear(1, r_embed)
        self.gru = nn.GRU(a_embed + s_embed + r_embed, hidden_size)  # running task summary
        self.fc_mu = nn.Linear(hidden_size, latent_dim)        # posterior mean
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)    # posterior log-variance

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)                # m = mu + sigma * eps

    def prior(self, batch_size):
        # initial belief before any data: hidden state zero -> readout
        h = torch.zeros(1, batch_size, self.hidden_size, device=self.fc_mu.weight.device)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.reparameterise(mu, logvar), mu, logvar, h

    def forward(self, actions, states, rewards, hidden_state=None, return_prior=True):
        """Sequences are [seq_len, batch, dim]. With return_prior=True the output
        length is seq_len+1: the prior, then q(m|tau_:t) for t = 1..seq_len."""
        if return_prior:
            prior_sample, prior_mu, prior_logvar, hidden_state = self.prior(actions.shape[1])
        ha = F.relu(self.action_encoder(actions))
        hs = F.relu(self.state_encoder(states))
        hr = F.relu(self.reward_encoder(rewards))
        h = torch.cat((ha, hs, hr), dim=-1)
        out, _ = self.gru(h, hidden_state)                     # one belief update per step
        mu, logvar = self.fc_mu(out), self.fc_logvar(out)      # posterior at every prefix t
        sample = self.reparameterise(mu, logvar)
        recurrent_state = out
        if return_prior:                                       # prepend the t=0 prior
            sample = torch.cat((prior_sample, sample))
            mu = torch.cat((prior_mu, mu))
            logvar = torch.cat((prior_logvar, logvar))
            recurrent_state = torch.cat((hidden_state, recurrent_state))
        return sample, mu, logvar, recurrent_state
```

The decoders are the latent-conditioned, task-shared reward and transition functions; they reconstruct the *whole* trajectory (future included) from an `m` inferred off a prefix:

```python
class RewardDecoder(nn.Module):
    """R'(r | s, a, s'; m): predict the reward from a transition, conditioned on
    the task latent. This is the auxiliary signal that forces m to encode the task."""

    def __init__(self, latent_dim, state_dim, action_dim, hidden=64,
                 s_embed=32, a_embed=16):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2 * s_embed + a_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, latent, prev_state, next_state, action):
        hps = F.relu(self.state_encoder(prev_state))
        hns = F.relu(self.state_encoder(next_state))
        ha = F.relu(self.action_encoder(action))
        return self.net(torch.cat((latent, hns, hps, ha), dim=-1))


class StateTransitionDecoder(nn.Module):
    """T'(s' | s, a; m): predict the next state from a transition and the task latent."""

    def __init__(self, latent_dim, state_dim, action_dim, hidden=64,
                 s_embed=32, a_embed=16):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + s_embed + a_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, latent, state, action):
        hs = F.relu(self.state_encoder(state))
        ha = F.relu(self.action_encoder(action))
        return self.net(torch.cat((latent, hs, ha), dim=-1))
```

The negative ELBO loss ties them together — encode the trajectory once to get a posterior at every prefix `t`, then for each `t` decode the *whole* trajectory under a sample from `q(m|tau_{:t})`, sum the reconstruction losses over all decode steps and over all `t`, and add the KL chained to the previous posterior:

```python
def gaussian_nll_mse(pred, target):
    # deterministic decoder -> squared-error reconstruction (a fixed-variance Gaussian NLL)
    return (pred - target).pow(2).mean(dim=-1)


def kl_chained_to_previous(mu, logvar):
    """KL terms for t=0..T. mu/logvar are [T+1, batch, latent]:
    index 0 is the empty-history posterior, regularised to N(0,I);
    later indices are regularised to the previous posterior."""
    unit_mu = torch.zeros_like(mu[:1])
    unit_logvar = torch.zeros_like(logvar[:1])
    all_mu = torch.cat((unit_mu, mu), dim=0)
    all_logvar = torch.cat((unit_logvar, logvar), dim=0)
    mu_t, logvar_t = all_mu[1:], all_logvar[1:]
    mu_p, logvar_p = all_mu[:-1], all_logvar[:-1]
    var_t, var_p = logvar_t.exp(), logvar_p.exp()
    kl = 0.5 * ((logvar_p - logvar_t)                  # log|S_{t-1}|/|S_t|
                + (var_t + (mu_p - mu_t).pow(2)) / var_p  # tr(S_{t-1}^-1 S_t) + Mahalanobis
                - 1.0).sum(dim=-1)                     # - K
    return kl


def compute_elbo_loss(encoder, reward_decoder, state_decoder,
                      prev_obs, next_obs, actions, rewards,
                      kl_weight=0.1, rew_coeff=1.0, state_coeff=1.0, decode_state=True):
    """One VAE update. Sequences are [T, batch, dim]; reconstruct past AND future."""
    # posterior at every prefix t (length T+1 incl. prior)
    _, mu, logvar, _ = encoder(actions, next_obs, rewards, hidden_state=None, return_prior=True)
    samples = encoder.reparameterise(mu, logvar)       # one m per ELBO term t
    T = next_obs.shape[0]

    rew_loss = state_loss = 0.0
    for t in range(samples.shape[0]):                  # sum_t ELBO_t
        m_t = samples[t].unsqueeze(0).expand(T, -1, -1)   # decode the WHOLE trajectory...
        rp = reward_decoder(m_t, prev_obs, next_obs, actions)  # ...incl. future steps
        rew_loss = rew_loss + gaussian_nll_mse(rp, rewards).sum(dim=0).mean()
        if decode_state:
            sp = state_decoder(m_t, prev_obs, actions)
            state_loss = state_loss + gaussian_nll_mse(sp, next_obs).sum(dim=0).mean()

    kl_loss = kl_chained_to_previous(mu, logvar).sum(dim=0).mean()  # sum_t KL_t
    return rew_coeff * rew_loss + state_coeff * state_loss + kl_weight * kl_loss
```

And the agent: the policy conditions on the state plus the *distribution* `(mu, logvar)` — not a sample — and the latent is detached so no RL gradient flows back through the encoder:

```python
class MetaRLAgent(nn.Module):
    """Encoder + decoders + policy. Policy sees state and the posterior (mu, logvar)
    so it can act on its own task uncertainty; the latent is detached from the RL loss."""

    def __init__(self, state_dim, action_dim, latent_dim=5, hidden=128):
        super().__init__()
        self.encoder = RNNEncoder(latent_dim, hidden, action_dim, state_dim)
        self.reward_decoder = RewardDecoder(latent_dim, state_dim, action_dim)
        self.state_decoder = StateTransitionDecoder(latent_dim, state_dim, action_dim)
        # policy input = state + (mu, logvar)  ->  2 * latent_dim of belief
        self.policy = build_policy(state_dim, 2 * latent_dim, action_dim, hidden)
        self.reset_belief(batch_size=1)

    def reset_belief(self, batch_size):
        _, self.mu, self.logvar, self.hidden = self.encoder.prior(batch_size)

    def update_belief(self, action, next_state, reward):
        # online: fold one transition into the running posterior (no prior prepended)
        _, mu, logvar, hidden = self.encoder(action[None], next_state[None], reward[None],
                                             hidden_state=self.hidden, return_prior=False)
        self.mu, self.logvar = mu[-1], logvar[-1]
        self.hidden = hidden[-1:].detach()

    def act(self, state, deterministic=False):
        belief = torch.cat((self.mu, self.logvar), dim=-1).detach()  # feed the posterior, detached
        return self.policy(torch.cat((state, belief), dim=-1), deterministic=deterministic)
```

Meta-training runs the two objectives with separate optimisers and buffers — the RL loss never touches the encoder:

```python
def meta_train(agent, task_distribution, rl_algorithm, num_iters,
               lr_vae=1e-3, kl_weight=0.1):
    vae_params = (list(agent.encoder.parameters())
                  + list(agent.reward_decoder.parameters())
                  + list(agent.state_decoder.parameters()))
    vae_optimiser = torch.optim.Adam(vae_params, lr=lr_vae)   # VAE: encoder + decoders only
    policy_buffer, vae_buffer = RecentBuffer(), TrajectoryBuffer()  # separate buffers

    for _ in range(num_iters):
        tasks = task_distribution.sample_batch()
        trajectories = rollout(agent, tasks)        # belief updated online during the rollout
        policy_buffer.add(trajectories); vae_buffer.add(trajectories)

        # RL: optimise return; latent is detached in agent.act, so encoder gets no RL grad
        rl_algorithm.update(agent.policy, policy_buffer.recent())

        # VAE: ELBO summed over prefixes t, reconstructing past AND future
        prev_obs, next_obs, actions, rewards = vae_buffer.sample()
        vae_optimiser.zero_grad()
        elbo_loss = compute_elbo_loss(agent.encoder, agent.reward_decoder, agent.state_decoder,
                                      prev_obs, next_obs, actions, rewards, kl_weight=kl_weight)
        elbo_loss.backward()
        vae_optimiser.step()
```

Let me trace the whole causal chain back. I started needing online, uncertainty-aware, scalable, unsupervised adaptation — the Bayes-optimal trade-off — and the exact specification, the BAMDP, was intractable in three places: unknown `R,T` parameterisation, intractable belief update, intractable belief-space planning. I knocked down the first by replacing the belief over `(R,T)` with a belief over a small learned latent `m` feeding a shared, task-conditioned reward/transition decoder. I knocked down the second with amortised variational inference: a recognition net `q_phi(m|tau_{:t})` giving a single-forward-pass posterior, trained by an ELBO I derived from the log-marginal via Jensen and the joint factorisation. The two moves that make it a *meta-RL* method rather than a plain VAE: encode the past but decode the *future* too, so the latent is forced to be a task descriptor and the agent learns to reason about unseen states; and train the ELBO at *every* context length, with each prior set to the previous posterior, so inference is online and the belief sharpens like a Bayes filter as data arrives. I knocked down the third — planning — by conditioning the policy directly on the posterior distribution `(mu, logvar)` and training it by RL, so it learns Bayes-optimal-style behaviour with no test-time planning; feeding the distribution rather than a sample is exactly what keeps it from collapsing back to inefficient posterior sampling. A recurrent (GRU) encoder gives the online, order-respecting belief update one step at a time; deleting the decoder and the ELBO recovers RL² as the special case, with the stochastic latent and the future-reconstruction auxiliary loss as the inductive biases that were missing. And I detach the latent from the RL loss, training the VAE and the policy with separate optimisers and buffers, for speed and to stop the two losses interfering. At test time it's forward passes only: the belief updates online, the policy acts on it, and adaptation is the posterior contracting — no gradients, no labels, no planning.
