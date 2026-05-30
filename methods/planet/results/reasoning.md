OK, let me think this through from scratch. I want an agent that controls something from raw pixels — locomotion, manipulation, a ball-in-cup — and the thing that actually hurts is not final performance, it's how many real environment episodes I have to burn. The strong model-free agents on these visual tasks chew through an enormous amount of interaction. Real steps are slow. So sample efficiency is the whole game, and I want to attack it at the root.

Here is the lever I keep coming back to. Planning is fantastic when I *know* the dynamics — model-predictive control on a simulated robot, tree search in a game with known rules. If I knew the transition function I could just search over action sequences, predict the reward each one earns, and pick the best. The only reason I can't do that here is that I don't have the simulator. So the obvious move: *learn* a dynamics model from the agent's own experience and plan inside it. And planning-in-a-learned-model has real advantages over model-free learning beyond data: I get a rich training signal because I'm predicting whole observations, not just bootstrapping a scalar value through Bellman backups; I can spend more compute at decision time to get better actions; and a dynamics model doesn't care about the reward function, so it could transfer across tasks. Good. The bet is clear. The hard part is going to be learning a model that's actually accurate enough to plan in.

Let me be honest about why that's hard, because every one of these is a landmine. A learned model is inaccurate, and worse, its errors *compound*: if each one-step prediction is a little off, a fifteen-step rollout drifts off into nonsense. The future is often genuinely uncertain — multiple things could happen — and a model that can only predict one future will be confidently wrong. And off the data it has seen, a naive model will extrapolate overconfidently. On top of all that, planning means scoring *thousands* of candidate action sequences at every single environment step, so whatever I build has to be cheap to roll forward.

Start with the observations: they're images, $64\times64\times3$. I am not going to predict forward in pixel space. It's enormous, slow, and most pixels are irrelevant to control — the wall texture behind the cheetah doesn't matter for running. I want a *latent* state. Encode each image into a compact vector $s_t$, predict forward in $s$-space, and only decode to pixels when I need a learning signal. The payoff is concrete: the planner can roll thousands of latent trajectories in parallel without ever rendering an image. So I want a latent state-space model — a transition $p(s_t\mid s_{t-1},a_{t-1})$, an observation model $p(o_t\mid s_t)$, and a reward model $p(r_t\mid s_t)$. This is structurally a non-linear Kalman filter / HMM with real-valued states, conditioned on actions and additionally predicting reward, which is exactly what lets it score action sequences without executing them.

One thing I should pin down about the setup. A single frame doesn't tell me the full state — I can't read velocity off one image of a cart-pole, and some tasks deliberately hide information. So this is a POMDP, not an MDP. The state $s_t$ is hidden; I have to *infer* it from the history of observations and actions. That has a consequence for how I'll do inference: I want a *filtering* posterior, $q(s_t\mid o_{\le t},a_{<t})$, that conditions only on the *past*. I could in principle use a smoothing posterior that also peeks at future observations during training, and it might give a tighter fit, but it would be the wrong object — at planning time the model has to predict the future *without* any future observations, so the inference distribution I train had better be the one that only ever looked backward. Filtering it is.

How do I train this thing? I don't observe $s_t$, so it's a latent-variable sequence model and the natural tool is a variational bound. I introduce an encoder that factorizes over time, $q(s_{1:T}\mid o_{1:T},a_{1:T})=\prod_t q(s_t\mid s_{t-1},a_{t-1},o_t)$ — a diagonal Gaussian whose mean and variance come from a conv net on the image followed by a small feedforward net. Let me actually derive the bound instead of just writing "ELBO," because I want to see exactly which terms fall out. The marginal likelihood of the observations, integrating out the latents, is
$$\ln p(o_{1:T}\mid a_{1:T})=\ln\,\mathbb E_{p(s_{1:T}\mid a_{1:T})}\!\Big[\prod_{t=1}^T p(o_t\mid s_t)\Big].$$
I can't sample the prior over latents and get anything useful — it never sees the data. So I importance-weight by my encoder: multiply and divide by $q$,
$$=\ln\,\mathbb E_{q(s_{1:T}\mid o_{1:T},a_{1:T})}\!\Big[\prod_{t=1}^T \frac{p(o_t\mid s_t)\,p(s_t\mid s_{t-1},a_{t-1})}{q(s_t\mid o_{\le t},a_{<t})}\Big].$$
Now $\ln$ of an expectation, and $\ln$ is concave, so Jensen pushes the log inside and gives me a lower bound:
$$\ge\,\mathbb E_q\Big[\sum_{t=1}^T \ln p(o_t\mid s_t)+\ln p(s_t\mid s_{t-1},a_{t-1})-\ln q(s_t\mid o_{\le t},a_{<t})\Big].$$
Group the last two logs at each $t$: the expectation over $q(s_t\mid\cdot)$ of $\ln p(s_t\mid s_{t-1},a_{t-1})-\ln q(s_t\mid\cdot)$ is exactly minus the KL from the encoder to the transition. So
$$\ln p(o_{1:T}\mid a_{1:T})\ \ge\ \sum_{t=1}^T\Big(\underbrace{\mathbb E_{q(s_t\mid o_{\le t},a_{<t})}[\ln p(o_t\mid s_t)]}_{\text{reconstruction}}-\underbrace{\mathbb E_{q(s_{t-1}\mid\cdot)}\big[\mathrm{KL}\big(q(s_t\mid o_{\le t},a_{<t})\,\|\,p(s_t\mid s_{t-1},a_{t-1})\big)\big]}_{\text{complexity}}\Big).$$
That's the whole objective. The reconstruction term wants the inferred state to explain the image; the complexity term is a one-step KL that pulls the *encoder's* guess for $s_t$ toward what the *transition* predicts from $s_{t-1}$. I estimate the outer expectations with a single reparameterized sample per step — write $s_t=\mu+\sigma\odot\epsilon$ so the sample is differentiable in the network outputs — and maximize by gradient ascent. The reward log-likelihood comes in by exact analogy, just another reconstruction term. And since I'll model the observation and reward as Gaussians with unit variance, $\ln p(o_t\mid s_t)$ is just a negative squared error up to a constant — the reconstruction terms are MSEs. The transition is a Gaussian with mean and variance from a feedforward net; the observation model is a Gaussian with mean from a deconv net and identity covariance; the reward is a scalar unit-variance Gaussian. Clean.

Now the form of the transition. My first instinct is the simplest thing: a fully stochastic latent Markov chain, $s_t\sim p(s_t\mid s_{t-1},a_{t-1})$, sample, repeat. Let me imagine rolling that forward for, say, fifteen steps for the planner. At step 1 I observe something important — the ball is to the left. For the model to *use* that at step 12, the information has to survive being resampled through a stochastic bottleneck at every intermediate step. Each step injects fresh noise and re-encodes. Purely stochastic transitions are terrible at carrying information forward reliably. In principle the model *could* learn to drive the variance to zero on the dimensions it wants to remember, turning those into a deterministic channel — but I have no reason to believe the optimizer will find that solution, and betting the whole method on a degenerate optimum it might not reach is a bad idea. So that's a wall: pure stochasticity can't remember.

The opposite extreme is a fully deterministic recurrent net — a GRU carrying $h_t=f(h_{t-1},a_{t-1})$. That remembers perfectly. But now there's no stochastic state at all, so the model can only ever predict *one* future. Faced with genuine uncertainty or multimodality it will average or commit, and it gives the variational machinery nothing — there's no sampled latent for a KL to regularize. So that's the other wall: pure determinism can't be uncertain.

Neither extreme works, and the two failures are complementary, so the resolution is to keep *both* paths. Split the state into a deterministic recurrent component $h_t$ that carries memory cleanly, and a stochastic component $s_t$ that captures what's genuinely uncertain:
$$h_t=f(h_{t-1},s_{t-1},a_{t-1})\quad[\text{a GRU}],\qquad s_t\sim p(s_t\mid h_t),$$
with the observation and reward models now reading off both pieces, $p(o_t\mid h_t,s_t)$ and $p(r_t\mid h_t,s_t)$, and the encoder folding the current image into the recurrent state, $q(s_t\mid h_t,o_t)$. The deterministic $h_t$ is the highway for long-range information; the stochastic $s_t$ is where multimodality lives and where the KL bites. This is the recurrent state-space model. There's one subtlety I have to get right: *all* information from the observation must pass through the sampling step of the encoder. If I let the image feed a deterministic path straight into the reconstruction, the model gets a shortcut — it can reconstruct $o_t$ by copying through $h_t$ without ever putting anything into the stochastic state, and then the prior I'm trying to learn (which has no access to $o_t$) learns nothing useful for prediction. So the encoder's only route from $o_t$ to the rest of the model is through the sampled $s_t$.

Let me get the encoder mechanics concrete, because the prior and posterior have to share structure. To compute the posterior at step $t$ I first run the deterministic update to get $h_t=f(h_{t-1},s_{t-1},a_{t-1})$ — this is identical to what the prior does. The prior then reads its mean and std off $h_t$ alone, $p(s_t\mid h_t)$. The posterior takes the *same* $h_t$, concatenates the observation embedding, and reads a different mean and std, $q(s_t\mid h_t,o_t)$. So the posterior is the prior with the observation glued on at the last layer — which is exactly what makes the KL meaningful: it measures how much the image moved the belief away from what the dynamics alone predicted. The thing I carry around and feed to the decoders and the planner is the concatenation $[\,s_t;\,h_t\,]$: stochastic part plus deterministic part.

Good — the model is settled. Now the policy. I said I'd plan, and I want to be precise that I am *not* learning a policy network or a value network at all. The policy *is* the planner: at each step, from my current belief over the state, search over future action sequences, predict the return of each one under the model, take the first action of the best plan, then re-plan at the next step with the new observation. That's model-predictive control, and re-planning each step is what lets the agent correct as new information comes in.

How do I do the search? The actions are continuous, the model is a black box I can only query forward, and I want robustness. A derivative-free, population-based optimizer fits: the cross-entropy method. Maintain a distribution over action sequences, sample a population, keep the elite fraction, refit the distribution to them, repeat. I'll represent the belief over the next $H$ actions as a *time-dependent* diagonal Gaussian, $a_{t:t+H}\sim\mathcal N(\mu_{t:t+H},\sigma^2_{t:t+H}\mathbb I)$ — one mean and variance per future time step — initialized at zero mean and unit variance. Each CEM iteration: sample $J$ candidate sequences from the current belief; evaluate each by rolling the model forward and summing predicted reward; sort, take the top $K$; and refit $\mu,\sigma$ to those $K$ elites. After $I$ iterations return $\mu_t$, the mean action for the *current* step only, and execute it.

Two details I want to reason about rather than assert. First, how many trajectories per candidate sequence? The honest thing would be to average several stochastic rollouts of the model per sequence to estimate its expected return. But CEM is already a population optimizer over many sequences — the population averaging does the variance reduction for me — so I'd rather spend the compute budget on evaluating *more distinct sequences* than on re-rolling each one. A single trajectory per sequence is enough. Second, and this is the one that bites if I get it wrong: should I warm-start the next step's CEM belief from this step's solution? It's tempting — it seems wasteful to start from $\mathcal N(0,I)$ every step. But warm-starting biases the search toward the previous plan and the optimizer gets stuck in local optima of the action landscape; resetting to zero mean and unit variance after every observation keeps the search exploratory. So I reset.

The thing that makes this cheap is that the reward is a function of the *latent* state. To score a candidate sequence I sample one state trajectory starting from my current state belief — $s_t$ from the posterior, then roll the *prior* $p(s_\tau\mid h_\tau)$ forward under the candidate actions — and sum the mean predicted rewards $\sum_\tau \mathbb E[p(r_\tau\mid s_\tau)]$. No image is ever generated during planning. So a whole batch of $J=1000$ sequences over a horizon of twelve is just a batch of cheap latent rollouts plus a feedforward reward head. That's the entire reason the latent model was worth building.

Now the experience loop, because the model and the data co-evolve and the agent can't visit everything up front. Start with a handful of seed episodes under random actions, train the model on them, and then alternate: do a fixed number of gradient updates, then go collect *one* more episode by planning with the current (partially trained) model, and add it to the dataset. When collecting, add small Gaussian noise to the planned action for exploration. And repeat each chosen action a fixed number of times before re-planning — action repeat. That last one isn't cosmetic: repeating actions shortens the effective horizon the planner has to reason over and gives the model a cleaner, slower-changing signal to predict, which matters a lot on these tasks. The model is trained on chunks of sequences sampled uniformly from the replay dataset.

So I have a complete agent. But let me stress-test the training objective, because I derived it for *one-step* predictions and the whole point is *multi-step* planning. Look back at the complexity term: the transition $p(s_t\mid s_{t-1},a_{t-1})$ only ever appears in a one-step KL against the posterior at the previous step. The gradient flows through $p(s_t\mid s_{t-1})$ into $q(s_{t-1})$, but it never traverses a *chain* of transitions — the model is never asked, during training, to predict two or three steps ahead through its own stochastic outputs. If the model had unlimited capacity this wouldn't matter: a perfect one-step predictor is automatically a perfect multi-step predictor. But with limited capacity and a restricted (Gaussian) family, the model that's best at one-step prediction is *not* in general the one that's best at multi-step prediction — and multi-step accuracy is exactly what the planner lives or dies by. This is a real gap in the objective.

So let me train the model on multi-step predictions directly, and — this is the part I want — do it without generating any extra images, purely in latent space. First I need a multi-step prior. Define the $d$-step prediction by repeatedly applying the transition and integrating out the intermediate states:
$$p(s_t\mid s_{t-d})\triangleq\int\prod_{\tau=t-d+1}^{t}p(s_\tau\mid s_{\tau-1})\,\mathrm d s_{t-d+1:t-1}=\mathbb E_{p(s_{t-1}\mid s_{t-d})}\big[p(s_t\mid s_{t-1})\big]$$
(dropping the action conditioning to keep the notation light; every state is implicitly conditioned on the actions). For $d=1$ this is just the one-step transition. Now I redo the variational bound but with the generative model using the $d$-step prior $p_d(s_{1:T})=\prod_t p(s_t\mid s_{t-d})$ in place of the one-step prior. Same importance-weighting-then-Jensen as before:
$$\ln p_d(o_{1:T})=\ln\mathbb E_q\Big[\prod_t \frac{p(o_t\mid s_t)\,p(s_t\mid s_{t-d})}{q(s_t\mid o_{\le t})}\Big]\ \ge\ \mathbb E_q\Big[\sum_t \ln p(o_t\mid s_t)+\ln p(s_t\mid s_{t-d})-\ln q(s_t\mid o_{\le t})\Big].$$
But the term $\ln p(s_t\mid s_{t-d})=\ln\mathbb E_{p(s_{t-1}\mid s_{t-d})}[p(s_t\mid s_{t-1})]$ is a log of an expectation again, and I'd rather not have to evaluate that marginal in closed form. So apply Jensen *a second time*, pushing the log inside that inner expectation:
$$\ln\mathbb E_{p(s_{t-1}\mid s_{t-d})}[p(s_t\mid s_{t-1})]\ \ge\ \mathbb E_{p(s_{t-1}\mid s_{t-d})}[\ln p(s_t\mid s_{t-1})].$$
Substituting and regrouping the $\ln p(s_t\mid s_{t-1})-\ln q(s_t\mid o_{\le t})$ into a KL gives the $d$-step bound:
$$\ln p_d(o_{1:T})\ \ge\ \sum_{t=1}^T\Big(\mathbb E_{q(s_t\mid o_{\le t})}[\ln p(o_t\mid s_t)]-\mathbb E_{p(s_{t-1}\mid s_{t-d})\,q(s_{t-d}\mid o_{\le t-d})}\big[\mathrm{KL}\big(q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})\big)\big]\Big).$$
Read it: the KL is still a one-step KL of posterior against one-step transition, but the *state it's conditioned on* is now drawn by rolling the prior forward $d$ steps from the posterior at $t-d$. So this term trains the model to make the stochastic transition land where the informed posterior is, even when it had to predict $d$ steps with no observations in between — which is precisely the situation the planner is in. And because the inner state is sampled, all the expectations are on the outside, so a single sample per term gives an unbiased estimator. No images generated.

There's a worry: by replacing the one-step prior with a $d$-step prior I've changed the generative model, so is $\ln p_d$ even a bound on the thing I care about, $\ln p(o_{1:T})$? Let me check with the data-processing inequality. The latent sequence is Markov, so for $d\ge1$ the further-apart states share *less* information: $\mathbb I(s_t;s_{t-d})\le\mathbb I(s_t;s_{t-1})$. Write the mutual information as $\mathbb H(s_t)-\mathbb H(s_t\mid s_{t-d})\le\mathbb H(s_t)-\mathbb H(s_t\mid s_{t-1})$, cancel $\mathbb H(s_t)$, flip signs (which reverses the inequality on the conditional entropies), and since the negative conditional entropy is $\mathbb E[\ln p(s_t\mid\cdot)]$ this gives $\mathbb E[\ln p(s_t\mid s_{t-d})]\le\mathbb E[\ln p(s_t\mid s_{t-1})]$. The $d$-step predictive log-likelihood is built from exactly these terms, so $\mathbb E[\ln p_d(o_{1:T})]\le\mathbb E[\ln p(o_{1:T})]$ in expectation over the dataset. So every bound on the $d$-step predictive distribution is also, in expectation, a bound on the one-step distribution — pushing up the multi-step objective pushes up the real likelihood too. Good, it's legitimate.

But a single $d$ only trains predictions of one distance, and for planning I need accuracy at *all* distances up to the horizon. So average the bound over $d=1$ through $D$, and — borrowing the per-term KL weighting from the $\beta$-VAE — attach a weight $\beta_d$ to each distance's KL:
$$\frac1D\sum_{d=1}^D\ln p_d(o_{1:T})\ \ge\ \sum_{t=1}^T\Big(\mathbb E_{q(s_t\mid o_{\le t})}[\ln p(o_t\mid s_t)]-\frac1D\sum_{d=1}^D\beta_d\,\mathbb E_{p(s_{t-1}\mid s_{t-d})\,q(s_{t-d}\mid o_{\le t-d})}\big[\mathrm{KL}\big(q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})\big)\big]\Big).$$
This is latent overshooting. It's a pure latent-space regularizer that forces consistency between one-step and multi-step predictions — which we just argued *should* be equivalent in expectation — and it costs nothing in image generation. One more thing I have to be careful about: for the overshooting distances $d>1$ I stop the gradient on the posterior distributions. I want the multi-step prediction to be trained *toward* the informed posterior, not to drag the posterior toward the sloppier multi-step prediction; stopping the gradient on the posterior side makes the regularizer one-directional, the way I intend it. I'll set all the $\beta_{>1}$ equal for simplicity, though one could tilt them toward long- or short-term.

It's worth noting where this lands for the actual agent: with the RSSM, the deterministic path already carries information forward cleanly enough that the one-step bound suffices in practice, so the final agent doesn't strictly need overshooting — but it's a general fix for latent sequence models whose stochastic path is being asked to remember, and it would help a model without the deterministic highway.

Let me now make all of this concrete in code, grounded in the structure of a real implementation. The model is the RSSM with its prior and posterior; the loss is the one-step variational bound with reconstruction MSEs and a KL that I clip below a few free nats (so the model isn't penalized for the first few nats of divergence — it spends its KL budget where it matters instead of being pushed to a degenerate match); and the agent is CEM in latent space.

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

# ---- encoder: image -> embedding (conv stack, from prior video models) ----
class Encoder(nn.Module):                       # 64x64x3 -> 1024
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3,   32,  4, stride=2)
        self.c2 = nn.Conv2d(32,  64,  4, stride=2)
        self.c3 = nn.Conv2d(64,  128, 4, stride=2)
        self.c4 = nn.Conv2d(128, 256, 4, stride=2)
    def forward(self, o):
        h = F.relu(self.c1(o)); h = F.relu(self.c2(h))
        h = F.relu(self.c3(h)); h = F.relu(self.c4(h))
        return h.reshape(h.size(0), -1)         # embedding

# ---- the contribution: RSSM = deterministic GRU path + stochastic latent ----
class RSSM(nn.Module):
    def __init__(self, state_dim=30, action_dim=1, rnn_dim=200,
                 hidden=200, min_std=0.1):
        super().__init__()
        self.rnn_dim, self.min_std = rnn_dim, min_std
        self.fc_input  = nn.Linear(state_dim + action_dim, hidden)   # -> GRU input
        self.gru       = nn.GRUCell(hidden, rnn_dim)                 # h_t = f(h,s,a)
        self.fc_prior  = nn.Linear(rnn_dim, hidden)                  # prior reads h
        self.prior_mu  = nn.Linear(hidden, state_dim)
        self.prior_std = nn.Linear(hidden, state_dim)
        self.fc_post   = nn.Linear(rnn_dim + 1024, hidden)           # posterior = h + obs
        self.post_mu   = nn.Linear(hidden, state_dim)
        self.post_std  = nn.Linear(hidden, state_dim)

    def prior(self, state, action, h):
        # advance deterministic path, then read stochastic state off h alone
        x = F.relu(self.fc_input(torch.cat([state, action], -1)))
        h = self.gru(x, h)                                          # deterministic memory
        f = F.relu(self.fc_prior(h))
        mu  = self.prior_mu(f)
        std = F.softplus(self.prior_std(f)) + self.min_std
        return Normal(mu, std), h

    def posterior(self, h, embed):
        # same h, but fold in the current observation -- ALL obs info goes
        # through this sampled latent, no deterministic shortcut to reconstruction
        f = F.relu(self.fc_post(torch.cat([h, embed], -1)))
        mu  = self.post_mu(f)
        std = F.softplus(self.post_std(f)) + self.min_std
        return Normal(mu, std)

# ---- decoders read BOTH parts of the state: features = [s_t ; h_t] ----------
class ObsModel(nn.Module):                      # p(o_t | s_t, h_t), Gaussian mean (deconv)
    def __init__(self, state_dim=30, rnn_dim=200):
        super().__init__()
        self.fc  = nn.Linear(state_dim + rnn_dim, 1024)
        self.d1  = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.d2  = nn.ConvTranspose2d(128,  64,  5, stride=2)
        self.d3  = nn.ConvTranspose2d(64,   32,  6, stride=2)
        self.d4  = nn.ConvTranspose2d(32,    3,  6, stride=2)
    def forward(self, s, h):
        x = self.fc(torch.cat([s, h], -1)).view(-1, 1024, 1, 1)
        x = F.relu(self.d1(x)); x = F.relu(self.d2(x)); x = F.relu(self.d3(x))
        return self.d4(x)                       # mean image (identity covariance)

class RewardModel(nn.Module):                   # p(r_t | s_t, h_t), scalar unit-variance Gaussian
    def __init__(self, state_dim=30, rnn_dim=200, hidden=200):
        super().__init__()
        self.f1 = nn.Linear(state_dim + rnn_dim, hidden)
        self.f2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, s, h):
        x = F.relu(self.f1(torch.cat([s, h], -1))); x = F.relu(self.f2(x))
        return self.out(x).squeeze(-1)          # mean reward (== MSE target)

# ---- one update: the one-step variational bound (recon MSE - free-nats KL) --
def model_loss(obs, act, rew, enc, rssm, obs_m, rew_m, free_nats=3.0):
    T, B = obs.shape[0], obs.shape[1]
    embed = enc(obs.reshape(-1, 3, 64, 64)).view(T, B, -1)
    s = obs.new_zeros(B, 30); h = obs.new_zeros(B, rssm.rnn_dim)
    states, hiddens, kl = [], [], 0.0
    for t in range(T - 1):
        prior, h = rssm.prior(s, act[t], h)        # p(s_{t+1}|h_{t+1})
        post     = rssm.posterior(h, embed[t + 1]) # q(s_{t+1}|h_{t+1},o_{t+1})
        s = post.rsample()                         # reparameterized sample
        states.append(s); hiddens.append(h)
        # complexity term: KL(posterior || prior), clipped below `free_nats`
        kl = kl + kl_divergence(post, prior).sum(-1).clamp(min=free_nats).mean()
    kl = kl / (T - 1)
    s_seq = torch.stack(states); h_seq = torch.stack(hiddens)
    recon = obs_m(s_seq.reshape(-1, 30), h_seq.reshape(-1, rssm.rnn_dim))
    recon = recon.view(T - 1, B, 3, 64, 64)
    # reconstruction terms = MSE (unit-variance Gaussian log-likelihood)
    obs_loss = 0.5 * F.mse_loss(recon, obs[1:], reduction='none').mean([0, 1]).sum()
    rew_pred = rew_m(s_seq.reshape(-1, 30), h_seq.reshape(-1, rssm.rnn_dim)).view(T - 1, B)
    rew_loss = 0.5 * F.mse_loss(rew_pred, rew[:-1])
    return obs_loss + rew_loss + kl             # maximize bound == minimize this

# ---- the policy IS the planner: CEM in latent space, replanned every step ---
def plan(belief_h, enc, rssm, rew_m, obs, action_dim,
         H=12, I=10, J=1000, K=100):
    embed = enc(obs)                                       # current frame
    post  = rssm.posterior(belief_h, embed)                # belief over current state
    mu  = obs.new_zeros(H, action_dim)                     # belief over action seq
    std = obs.new_ones(H, action_dim)                      #   start N(0, I)
    for _ in range(I):
        cand = Normal(mu, std).sample([J]).transpose(0, 1) # (H, J, action_dim)
        s = post.sample([J]).squeeze(1)                    # one latent traj per candidate
        h = belief_h.repeat(J, 1)
        ret = obs.new_zeros(J)
        for t in range(H):                                 # roll the PRIOR forward in latent space
            prior, h = rssm.prior(s, cand[t], h)
            s = prior.sample()
            ret = ret + rew_m(s, h)                         # sum predicted reward, never render
        elite = ret.argsort(descending=True)[:K]           # keep top K
        top   = cand[:, elite, :]
        mu  = top.mean(1)
        std = (top - mu.unsqueeze(1)).abs().sum(1) / (K - 1)  # refit belief to elites
    return mu[0]                                           # execute only the first action
```

So the causal chain, start to finish: I want to control from pixels with few real episodes, and planning is the data-efficient route — *if* I can learn a model worth planning in. Pixel-space prediction is hopeless, so I move to a compact latent state and a variational bound that infers it with a backward-only filtering posterior. A purely stochastic latent can't remember and a purely deterministic recurrence can't be uncertain, so the state splits into a GRU memory highway plus a stochastic latent — the RSSM — with all observation information forced through the sampled latent. Because reward is a function of that latent, a derivative-free CEM planner can score a thousand action sequences per step entirely in latent space, re-planning each step under MPC and never rendering an image, which is the whole reason the latent model pays off. The one-step bound only trains one-step transitions, so I generalize it: a $d$-step prior, a second Jensen step to keep it tractable, a data-processing-inequality argument that the multi-step bound still bounds the true likelihood, and an average over distances with $\beta_d$ weights — latent overshooting — that trains accurate multi-step predictions purely in latent space, which is exactly what the planner needs.
