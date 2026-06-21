An agent dropped into an environment it has never seen — drawn from a known family of related environments that share state and action spaces but differ in their reward and transition functions — is graded on the return it accumulates *while it is still figuring the environment out*, from the very first steps, not after some long separate adaptation phase. Every action is then a gamble between probing to learn what kind of environment this is and cashing in on what it already suspects, and I want the agent to explore exactly as much as it is worth given how much time remains, and not a step more. The clean specification of "exactly as much as it is worth" is the Bayes-Adaptive MDP: put a prior $b_0 = p(R,T)$ over the unknown reward and transition functions, carry a posterior $b_t = p(R,T\mid\tau_{:t})$ as experience $\tau_{:t}=(s_0,a_0,r_1,s_1,\dots,s_t)$ comes in, glue this belief onto the environment state to form a hyper-state $s^+_t = (s_t, b_t)$, and maximise $J^+(\pi) = \mathbb{E}[\sum_t \gamma^t R^+]$ over a horizon $H^+$ (often $H^+ = N\times H$, spanning the first $N$ episodes, since whether a costly probe pays off depends on how long one gets to exploit the answer). The maximiser is the Bayes-optimal policy, which seeks information only insofar as resolving uncertainty raises expected return within the horizon. The specification is perfect, but solving the BAMDP is hopeless past toy size in three distinct places: I usually do not know how to *parameterise* the true $R,T$; the belief update $p(R,T\mid\tau_{:t})$ is itself intractable; and even handed the posterior, planning in belief space is intractable. Tabular Bayesian-RL planners breach none of these at scale — they need a hand-built prior and belief update and expensive tree search, and live in small discrete spaces. Posterior sampling dodges only the planning wall by drawing one hypothesis MDP per episode and chasing it, which explores inefficiently — it commits to one goal at a time, walks there, finds nothing, resamples, revisits cells — and still needs the posterior. Among meta-learning tools, RL² makes the policy a recurrent net fed the observation plus the previous action, reward, and done flag, and trains by ordinary RL so the inner learner is distilled into the recurrence; but its hidden state is opaque, with no explicit representation of the task and crucially none of *uncertainty*, and it grows unstable across episode resets. MAML separates exploration (rollouts under the un-adapted policy) from exploitation (after the gradient steps), so exploration is never optimised for online return. PEARL's permutation-invariant encoder treats context as an unordered set, throwing away the temporal structure of exploration, and acts by sampling a latent — posterior sampling again. The one method that conditions on a posterior, supervised task-inference, needs privileged task labels. So the gap is fourfold: I want the policy to condition on task *uncertainty*, the inference to be *online* and to respect the sequential order of experience, the whole thing *tractable and scalable* to deep nets and continuous control, and *unsupervised*. No existing method has all four.

I propose VariBAD — variational Bayes-Adaptive deep RL — which knocks down the three intractability walls in turn. For the first, I refuse to parameterise the true $R,T$ in their native million-parameter form. The tasks share structure and differ only in something low-dimensional — a goal position, a target velocity — so I introduce a learned, low-dimensional stochastic latent $m$ that stands in for the task and write $R_i(r\mid s,a,s')\approx R(r\mid s,a,s';m_i)$ and $T_i(s'\mid s,a)\approx T(s'\mid s,a;m_i)$, where $R$ and $T$ are now single networks *shared across all tasks* and modulated by $m$. This collapses the belief over $(R,T)$ into a belief over the small vector $m$, and the shared decoders learn from every task's data instead of being re-estimated per task. For the second wall I use amortised variational inference: a recognition network $q_\phi(m\mid\tau_{:t})$ maps the experience so far to an approximate posterior, trained jointly with a generative model of the trajectory. The model objective $\max_{\theta,\phi}\mathbb{E}_\rho[\log p_\theta(\tau)]$ is intractable because it marginalises $m$ out, but the variational move turns it into something optimisable. Slipping the recognition density in as a ratio equal to one and applying Jensen's inequality (the log is concave) gives
$$\log p_\theta(\tau) = \log \mathbb{E}_{q_\phi(m\mid\tau_{:t})}\!\left[\frac{p_\theta(\tau,m)}{q_\phi(m\mid\tau_{:t})}\right] \ge \mathbb{E}_{q_\phi(m\mid\tau_{:t})}\!\left[\log\frac{p_\theta(\tau,m)}{q_\phi(m\mid\tau_{:t})}\right],$$
and factoring $p_\theta(\tau,m)=p_\theta(\tau\mid m)\,p_\theta(m)$ and splitting the log yields the evidence lower bound
$$\mathrm{ELBO}_t = \mathbb{E}_{q_\phi(m\mid\tau_{:t})}\!\left[\log p_\theta(\tau_{:H^+}\mid m)\right] - \mathrm{KL}\!\left(q_\phi(m\mid\tau_{:t})\,\|\,p_\theta(m)\right),$$
a reconstruction term plus a KL pulling the posterior toward a prior, estimated by Monte Carlo with the reparameterisation trick $m = \mu + \sigma\cdot\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$. The posterior is now a single forward pass — fast enough to run online at every timestep. The third wall, planning, I dissolve rather than climb: I condition the policy directly on the posterior and train it end-to-end by RL, so it learns at meta-training time to act well as a function of its current belief, with no test-time planning at all.

What makes this a meta-RL method rather than a plain VAE comes down to three design choices, each of which beats the obvious alternative. First, what to reconstruct. A conventional VAE encodes its input and reconstructs that same input; if I encode $\tau_{:t}$ and decode $\tau_{:t}$, the latent only needs to compress the history already seen. I instead encode only the past — the encoder can never see more than $\tau_{:t}$ at decision time $t$ — but decode the *whole* trajectory $\tau_{:H^+}$, future included, since at training time I have the full rollout. Predicting the future from an $m$ inferred off a prefix is impossible unless $m$ has captured the underlying task, so this forces $m$ to be a task descriptor and teaches the agent to reason about unseen states. The reconstruction factorises over time,
$$\log p(\tau_{:H^+}\mid m, a) = \log p(s_0\mid m) + \sum_i\big[\log p(s_{i+1}\mid s_i,a_i,m) + \log p(r_{i+1}\mid s_i,a_i,s_{i+1},m)\big],$$
so the decoder is exactly two heads sharing $m$ — a transition decoder $T'(s_{i+1}\mid s_i,a_i;m)$ and a reward decoder $R'(r_{i+1}\mid s_i,a_i,s_{i+1};m)$, the latent-conditioned environment model. Second, *which* context length to train on. Training the ELBO only at the final step never teaches the encoder to give a good posterior early, precisely when exploration decisions matter most, so I train the ELBO at *all* context lengths and sum them, giving the full objective
$$L(\phi,\theta,\psi) = \mathbb{E}_{p(M)}\!\left[J(\psi,\phi) + \lambda\sum_{t=0}^{H^+}\mathrm{ELBO}_t(\phi,\theta)\right],$$
where $\lambda$ weights the supervised model term against the RL return $J$ because $\phi$ is shared. The encoder then learns online inference: a useful $q(m\mid\tau_{:t})$ for every $t$, contracting smoothly as data arrives. Third, the prior in each KL. A fixed $\mathcal{N}(0,I)$ at every step pulls the belief back toward total ignorance at every step, fighting the contraction I want. What I actually want is a Bayesian filter: regularise the belief at step $t$ toward the belief at step $t-1$, not toward the origin. So I set the prior in $\mathrm{ELBO}_t$ to the previous posterior $q_\phi(m\mid\tau_{:t-1})$, with only the empty-history posterior anchored to $\mathcal{N}(0,I)$. With diagonal Gaussians $\mathcal{N}(\mu_t,S_t)$ and previous $\mathcal{N}(\mu_{t-1},S_{t-1})$ this is the full Gaussian-to-Gaussian divergence
$$\mathrm{KL}\!\left(\mathcal{N}(\mu_t,S_t)\,\|\,\mathcal{N}(\mu_{t-1},S_{t-1})\right) = \tfrac{1}{2}\Big[\log\tfrac{|S_{t-1}|}{|S_t|} - K + \mathrm{tr}(S_{t-1}^{-1}S_t) + (\mu_{t-1}-\mu_t)^\top S_{t-1}^{-1}(\mu_{t-1}-\mu_t)\Big],$$
with $K$ the latent dimension; this chain penalises changing one's mind without evidence. (If instead every posterior is pulled to the fixed unit Gaussian, this reduces to $\mathrm{KL}(q\|\mathcal{N}(0,I)) = -\tfrac{1}{2}\sum_k(1 + \log\sigma_k^2 - \mu_k^2 - \sigma_k^2)$.)

The encoder must compute the posterior online, folding in one transition at a time, and respect the *order* of experience — which rules out a permutation-invariant set encoder and makes recurrence the right primitive. So I embed each $(a,s,r)$ with small feature extractors, concatenate, run a GRU whose hidden state carries the running summary, and read out $\mu$ and $\log\sigma^2$ from the hidden state. One GRU step per environment step is one belief update per step, and feeding the whole trajectory through in one pass gives the posterior at every prefix length at once — exactly the $q(m\mid\tau_{:t})$ for all $t$ the summed ELBO needs. The hidden state persists across episode boundaries within a task (the BAMDP horizon spans several rollouts) and is zeroed only when a fresh task begins. Deleting the decoder and the entire ELBO leaves a recurrent policy fed $(s,a,r)$ trained by RL — that is RL². The differences that remain are precisely the inductive biases I argued for: a *stochastic* latent that can represent uncertainty (an opaque hidden vector cannot say "I am not sure yet," and being able to say so drives early exploration), and the decoder reconstructing past and future transitions and rewards. For the policy I resist the obvious-but-wrong choice of sampling $m\sim q(m\mid\tau_{:t})$ and feeding the sample — that is posterior sampling, acting as if one draw were the truth, and reproduces the inefficient hypothesis-chasing I want to beat. Instead I condition the policy on the posterior *itself*, both mean and variance, $\pi(a_t\mid s_t, q(m\mid\tau_{:t}))$, implemented by concatenating $(\mu_t,\log\sigma_t^2)$ with the state, so the policy can see its own uncertainty: probe when the variance is high, exploit when it is low. This is the BAMDP hyper-state $(s_t,b_t)$ made concrete with the learned latent belief playing the role of $b_t$. Finally, a practical choice that matters: I do not backprop the RL gradient through the encoder. Letting it flow would force recomputing the GRU embeddings on every on-policy minibatch pass (expensive) and would make the RL and VAE losses fight over $\phi$ (interference requiring careful $\lambda$ tuning). Stopping the RL gradient at the encoder lets the decoder's reconstruction alone shape $m$ — rich enough on its own — and lets the embeddings be reused; so I detach the latent before the policy and train the VAE and the policy with separate optimisers and buffers. (Per domain I can simplify further: on the continuous-control families a reward-only decoder, dropping the transition head, works well even where the dynamics change, since the reward usually carries the task identity.) At test time the payoff is clean — roll out on a held-out task with forward passes only: feed each new $(s,a,r)$ into the encoder to update the posterior, hand state-plus-posterior to the policy, act, repeat; no gradient steps, no inner-loop adaptation, no privileged task label, with adaptation being the belief sharpening as experience accrues. Defaults for the MuJoCo families are PPO (A2C in the gridworld), latent dim $5$, KL weight $\lambda_{\mathrm{KL}}=0.1$, policy LR $7\times10^{-4}$, VAE LR $1\times10^{-3}$, GRU hidden size $128$, state/action/reward embeddings of $32/16/16$, and a two-hidden-layer reward decoder with MSE (deterministic-Gaussian) reconstruction.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    """q_phi(m | tau_:t): embed (a,s,r), run a GRU online, read out (mu, logvar)."""

    def __init__(self, latent_dim=5, hidden_size=128, action_dim=2, state_dim=2,
                 a_embed=16, s_embed=32, r_embed=16):
        super().__init__()
        self.latent_dim, self.hidden_size = latent_dim, hidden_size
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.reward_encoder = nn.Linear(1, r_embed)
        self.gru = nn.GRU(a_embed + s_embed + r_embed, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def prior(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size, device=self.fc_mu.weight.device)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.reparameterise(mu, logvar), mu, logvar, h

    def forward(self, actions, states, rewards, hidden_state=None, return_prior=True):
        if return_prior:
            prior_sample, prior_mu, prior_logvar, hidden_state = self.prior(actions.shape[1])
        h = torch.cat((F.relu(self.action_encoder(actions)),
                       F.relu(self.state_encoder(states)),
                       F.relu(self.reward_encoder(rewards))), dim=-1)
        out, _ = self.gru(h, hidden_state)
        mu, logvar = self.fc_mu(out), self.fc_logvar(out)
        sample = self.reparameterise(mu, logvar)
        recurrent_state = out
        if return_prior:
            sample = torch.cat((prior_sample, sample))
            mu = torch.cat((prior_mu, mu))
            logvar = torch.cat((prior_logvar, logvar))
            recurrent_state = torch.cat((hidden_state, recurrent_state))
        return sample, mu, logvar, recurrent_state


class RewardDecoder(nn.Module):
    """R'(r | s, a, s'; m): the auxiliary signal that forces m to encode the task."""

    def __init__(self, latent_dim, state_dim, action_dim, hidden=64, s_embed=32, a_embed=16):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2 * s_embed + a_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, latent, prev_state, next_state, action):
        h = torch.cat((latent,
                       F.relu(self.state_encoder(next_state)),
                       F.relu(self.state_encoder(prev_state)),
                       F.relu(self.action_encoder(action))), dim=-1)
        return self.net(h)


class StateTransitionDecoder(nn.Module):
    """T'(s' | s, a; m)."""

    def __init__(self, latent_dim, state_dim, action_dim, hidden=64, s_embed=32, a_embed=16):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + s_embed + a_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim))

    def forward(self, latent, state, action):
        h = torch.cat((latent, F.relu(self.state_encoder(state)),
                       F.relu(self.action_encoder(action))), dim=-1)
        return self.net(h)

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
    return 0.5 * ((logvar_p - logvar_t)
                  + (var_t + (mu_p - mu_t).pow(2)) / var_p - 1.0).sum(dim=-1)


def compute_elbo_loss(encoder, reward_decoder, state_decoder,
                      prev_obs, next_obs, actions, rewards,
                      kl_weight=0.1, rew_coeff=1.0, state_coeff=1.0, decode_state=True):
    _, mu, logvar, _ = encoder(actions, next_obs, rewards, hidden_state=None, return_prior=True)
    samples = encoder.reparameterise(mu, logvar)
    T = next_obs.shape[0]
    rew_loss = state_loss = 0.0
    for t in range(samples.shape[0]):                 # sum_t ELBO_t
        m_t = samples[t].unsqueeze(0).expand(T, -1, -1)   # decode whole trajectory incl. future
        rp = reward_decoder(m_t, prev_obs, next_obs, actions)
        rew_loss = rew_loss + (rp - rewards).pow(2).mean(-1).sum(0).mean()
        if decode_state:
            sp = state_decoder(m_t, prev_obs, actions)
            state_loss = state_loss + (sp - next_obs).pow(2).mean(-1).sum(0).mean()
    kl_loss = kl_chained_to_previous(mu, logvar).sum(0).mean()
    return rew_coeff * rew_loss + state_coeff * state_loss + kl_weight * kl_loss

class VariBADAgent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=5, hidden=128):
        super().__init__()
        self.encoder = RNNEncoder(latent_dim, hidden, action_dim, state_dim)
        self.reward_decoder = RewardDecoder(latent_dim, state_dim, action_dim)
        self.state_decoder = StateTransitionDecoder(latent_dim, state_dim, action_dim)
        self.policy = build_policy(state_dim, 2 * latent_dim, action_dim, hidden)  # state + (mu,logvar)
        self.reset_belief(1)

    def reset_belief(self, batch_size):
        _, self.mu, self.logvar, self.hidden = self.encoder.prior(batch_size)

    def update_belief(self, action, next_state, reward):
        _, mu, logvar, hidden = self.encoder(action[None], next_state[None], reward[None],
                                             hidden_state=self.hidden, return_prior=False)
        self.mu, self.logvar = mu[-1], logvar[-1]
        self.hidden = hidden[-1:].detach()

    def act(self, state, deterministic=False):
        belief = torch.cat((self.mu, self.logvar), dim=-1).detach()   # posterior, detached from RL
        return self.policy(torch.cat((state, belief), dim=-1), deterministic=deterministic)


def meta_train(agent, task_distribution, rl_algorithm, num_iters, lr_vae=1e-3, kl_weight=0.1):
    vae_params = (list(agent.encoder.parameters())
                  + list(agent.reward_decoder.parameters())
                  + list(agent.state_decoder.parameters()))
    vae_optimiser = torch.optim.Adam(vae_params, lr=lr_vae)
    policy_buffer, vae_buffer = RecentBuffer(), TrajectoryBuffer()
    for _ in range(num_iters):
        trajectories = rollout(agent, task_distribution.sample_batch())   # belief updated online
        policy_buffer.add(trajectories); vae_buffer.add(trajectories)
        rl_algorithm.update(agent.policy, policy_buffer.recent())         # no RL grad to encoder
        prev_obs, next_obs, actions, rewards = vae_buffer.sample()
        vae_optimiser.zero_grad()
        loss = compute_elbo_loss(agent.encoder, agent.reward_decoder, agent.state_decoder,
                                 prev_obs, next_obs, actions, rewards, kl_weight=kl_weight)
        loss.backward()
        vae_optimiser.step()
```
