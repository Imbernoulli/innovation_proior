OK, let me think this through from scratch. I want an agent that acts from raw pixels — drive a car around a randomly generated track, dodge fireballs in a first-person room — and I have a strong hunch that the agent would be far better off if it carried a big, expressive predictive model of its world. A large recurrent net could learn rich spatial and temporal structure: what the scene looks like, how it evolves, what my actions do to it. The trouble is purely one of training. If I try to learn the millions of weights of a big network *directly from reward*, I hit the credit-assignment problem head on — reward is sparse and delayed, and I have to figure out which of millions of parameters deserves blame or credit for an outcome many steps later. That's why model-free RL in practice uses small networks: they iterate fast to a decent policy precisely because there isn't much to assign credit to. But small means low capacity.

Let me stare at that tension, because it suggests its own resolution. The credit-assignment difficulty is tied to *reward-driven* optimization. But predicting the next frame from the current frame and action is not reward-driven at all — it's a dense, self-supervised signal available at every single time step, and I can train it with ordinary backpropagation with no credit-assignment problem whatsoever. So what if I put almost all the capacity into a model trained *that* way, and leave only a sliver of parameters to be optimized against reward? Split the agent: a large **world model**, learned unsupervised from observed transitions, and a small **controller**, the only thing that ever sees reward. If the controller is tiny — a few hundred parameters — then the hard reward-driven search happens in a tiny space, and I might not even need gradients for it; a derivative-free optimizer could handle it. Meanwhile the capacity lives safely in the world model where training is easy. That's the bet.

Now I have to design the world model. The agent's experience is a stream of images, and two distinct things need compressing: *what I see at each instant*, and *how things change over time*. Those are different jobs, so let me give them to different modules.

Take the spatial one first. Each frame is high-dimensional pixels; I want a compact code. An autoencoder would do it, but a plain autoencoder gives me an arbitrary, possibly jagged latent space — and that matters because, as I'll see in a moment, I'm going to *predict and sample* in this latent space, which means I'll be feeding the model latent vectors that it didn't see exactly during training. So I want the latent space to be smooth and bounded, and I want each code to carry limited information so it can't memorize idiosyncratic pixel detail. A variational autoencoder gives me both: the encoder outputs the mean and standard deviation of a diagonal Gaussian, I sample $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$, the decoder reconstructs the frame, and the loss is the reconstruction error plus a KL term pulling the posterior toward a standard-normal prior. The KL is doing real work for me here: it caps the information capacity of each code and, by anchoring the latent to a unit Gaussian, makes the space robust to the slightly-off latent vectors that my predictive model will inevitably produce. Call this the vision module V: frame $\to z$.

Now the temporal one. I want to compress *what happens over time* — given where I am and what I do, what will the next latent look like? The obvious thing is a recurrent net that maps the current latent and action to the next latent, carrying a hidden state $h_t$ forward: model $z_{t+1}$ as a function of $a_t, z_t, h_t$. But should it predict a single $z_{t+1}$? Let me think about whether that's enough. These environments are stochastic in their *consequences*: in the dodging game a monster might or might not fire, and those are genuinely different futures. A net that outputs one deterministic $z_{t+1}$ has to average over those futures, which produces a blurry, incoherent prediction — the average of "fireball here" and "no fireball" is neither. So I don't want a point prediction; I want the *distribution* over the next latent. And it had better be able to represent *multiple modes*, because "fireball" and "no fireball" are distinct outcomes, not a spread around one mean.

A mixture density network is exactly built for this: instead of one output it emits the parameters of a mixture of Gaussians — mixing weights $\pi_k$, means $\mu_k$, standard deviations $\sigma_k$ — and so it can put probability mass on several distinct futures at once. Wire it onto the recurrent net's hidden state and I have an MDN-RNN that models
$$P(z_{t+1}\mid a_t, z_t, h_t)=\sum_{k=1}^K \pi_k(h_t)\,\mathcal N\big(z_{t+1}\mid \mu_k(h_t),\,\sigma_k(h_t)\big),$$
trained by minimizing the negative log-likelihood of the observed next latent under this mixture. I'll keep the per-component Gaussians factored (diagonal, no cross-correlation between latent dimensions) — it's enough, since the multimodality I care about is captured by *which mixture component fires*, not by correlations within a component. Notice the asymmetry with V: V's latent is a single diagonal Gaussian per frame and that's fine, because a single frame isn't ambiguous in the same way; the *dynamics* are where discrete, multimodal randomness lives, so the mixture goes on the temporal model M. And I'll borrow one more thing from sequence-generation work — a **temperature** $\tau$ that scales the standard deviations (and softens the mixture weights) at sampling time, so I can dial how uncertain the model's generated futures are. I don't have a use for that knob yet, but I suspect I will.

Let me be careful about training M so it doesn't lock onto noise. The latents $z$ are themselves *samples* from V's posterior, so each frame doesn't have one fixed $z$ — it has a $\mu,\sigma$. If I train M on one fixed sampled $z$ per frame it could overfit that particular draw. So I store the precomputed $\mu,\sigma$ for each frame and re-sample $z\sim\mathcal N(\mu,\sigma)$ fresh each time I build a training batch. Cheap, and it keeps M honest about V's uncertainty.

Now the controller C. I want it tiny, so the reward-driven search is easy. The simplest possible thing is a single linear layer mapping its input features to the action,
$$a_t = W_c\,[\,z_t\,;\,h_t\,] + b_c,$$
with a $\tanh$ to bound the action into the valid range. The question that actually matters is *what features it sees*. The obvious choice is just the current latent $z_t$ — that's the present. But $z_t$ is only a snapshot; it has no notion of where things are heading. And I already have, sitting right there in M, a vector that summarizes the future: the recurrent hidden state $h_t$, from which M reads off the whole distribution over $z_{t+1}$. So $h_t$ encodes "what to expect next." If I feed C *both* $z_t$ and $h_t$, the controller gets the present *and* a compressed view of the predicted future, and it can act reflexively — no need to explicitly roll the model forward and search over imagined trajectories at decision time, the way a planner would. The hidden state already did that work; C just queries it. I expect that giving C only $z_t$ produces twitchy, shortsighted control (it can't anticipate a corner), and adding $h_t$ smooths it out. That's the whole reason M's hidden state is the right feature and not just M's point prediction.

The rollout, then, is a tight loop. Reset the environment, initialize M's hidden state. Each step: V encodes the frame to $z_t$; C maps $[z_t;h_t]$ to an action $a_t$; the environment advances and returns the next frame and reward; M consumes $(a_t, z_t, h_t)$ to produce the next hidden state $h_{t+1}$. The cumulative reward of this loop is C's fitness.

How do I optimize C? Because C is a few hundred parameters and the environment is a black box I can only roll out, I don't want gradients through the environment at all — and I can't get them through the real game engine anyway. A derivative-free evolution strategy is the right tool: it needs only the *final* cumulative reward of a rollout, it's robust, and it parallelizes trivially because each candidate controller is an independent rollout. CMA-ES specifically adapts a search covariance and works well up to a few thousand parameters, which is exactly C's regime. So: maintain a population of candidate parameter vectors, run each one for several rollouts with different random seeds, set its fitness to the *average* cumulative reward over those rollouts (averaging tames the variance from random track / random starts), and let CMA-ES update the search distribution toward the good ones.

That's a complete agent, and it should already work on the racing task: V and M are trained once on a dataset of random-policy rollouts (V to reconstruct frames, M to predict the next latent), and then CMA-ES evolves the linear C against the real environment using $[z_t;h_t]$ as features.

Now the more ambitious idea, the one the whole architecture has been quietly setting up. M models $P(z_{t+1}\mid a_t,z_t,h_t)$. That is *exactly the transition function of an environment* in latent space. If M also predicts whether the episode ends — add a head that outputs the probability of a "done" event $d_t$ — then M has everything a real environment interface has: given a latent state and an action, it produces the next latent state and a done flag. So I can wrap M as if it were the environment itself and train C *entirely inside M's hallucination* — sample $z_{t+1}$ from M's mixture, feed it back in, never touch a real frame, never even run V during the dream because I'm already living in latent space. The controller learns in a dream and then I transfer the learned $W_c, b_c$ back to the real environment. (To make $d_t$ stable I'll threshold it — call it done when the predicted probability exceeds one-half — rather than sampling a Bernoulli every step, since death is a rare per-step event and sampling it would be noisy.)

This is appealing — the dream has no rendering cost, no game engine, it's pure latent rollouts — but I should immediately distrust it, because there's an obvious failure mode and it's the same one that has dogged learned-model agents for decades. M is only an *approximation* of the real dynamics. A controller optimized against M with a powerful black-box search will not politely stay where M is accurate; it will hunt for *whatever* makes M's predicted reward high, including trajectories that exploit M's mistakes. Concretely, I'd expect to see the controller discover an adversarial policy — moving in some peculiar way so that, in the dream, the monsters never fire at all, or any forming fireball quietly vanishes. It would score perfectly in the dream and then fall apart in reality, because it's living in states M never modeled correctly. And it's worse than for a planner that only queries M occasionally: by training C *inside* M I'm handing the controller direct access to all of M's hidden states — essentially the internals of the simulator — so it can learn to drive those internals into convenient corners. This is precisely why so much prior work learns a dynamics model but stops short of fully *replacing* the real environment with it: a deterministic model is trivially exploitable.

So how do I stop the controller from cheating the dream? Here's where the stochastic M earns its keep. Because M outputs a *distribution* over futures — a mixture, with real probability mass on alternatives — the dream is not a single deterministic track the controller can game; it's noisy. And I have a dial for exactly how noisy: the temperature $\tau$ I built into the sampler. Crank $\tau$ up and M's sampled futures spread out and jump between mixture modes more freely; the fireballs come at less predictable times, the agent occasionally dies for no clean reason. An adversarial policy that depended on M behaving deterministically can't survive that randomness, so the controller is forced to learn a robust policy instead of an exploit. This is the payoff of having made M a mixture density model rather than a deterministic predictor, and it's why the mixture's discrete modes matter — a single Gaussian couldn't represent "fireball vs. no fireball," so it couldn't generate the discrete random events that make the dream un-gameable.

There's a sweet spot, though, and I can reason out both failure ends. Push $\tau$ too low — say $0.1$ — and M collapses toward a single deterministic mode: in the dodging game the monsters never manage to fire at all because M can't jump to the mixture component where a fireball forms, so the dream becomes trivially winnable and any policy I learn there scores perfectly and then dies instantly in reality, underperforming even random. Push $\tau$ too high and the dream becomes so chaotic there's no learnable signal — the agent can't tell good actions from bad through the noise. So $\tau$ is a genuine hyperparameter trading realism against exploitability, tuned a bit above $1$. And the diagnostic I'd trust: a controller that performs well at *high* temperature, where the dream is hard and un-exploitable, should transfer to the real (cleaner, less noisy) environment well — surviving a harsher nightmare should make reality easy.

Let me also handle the case where one pass isn't enough. On the simple tasks, a world model trained on random-policy data is fine, because random actions already visit the relevant states. But on a harder task the interesting parts of the world only become reachable *after* the agent has learned to get there, so a random-policy dataset would never show M those regions and the dream would be blank exactly where it matters. The fix is an iterative loop: initialize M and C; roll out the *current* C in the real environment and log the actions and observations; retrain M on the enlarged dataset — and now I'd have M predict not just the next observation and done but also the next action and reward, so M can absorb skills the controller has already learned; retrain C inside the improved M; repeat. And there's a natural exploration signal hiding in M's own loss: where M predicts poorly, the agent is in unfamiliar territory, so flipping the sign of M's loss into a reward pushes C to seek out exactly the experience that would most improve the world model.

Let me now write the pieces concretely, grounded in a real implementation. V is the convolutional VAE; M is the LSTM with the mixture-density head plus reward and done heads; C is the single linear layer; the controller is trained by CMA-ES, optionally inside M.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---- V: convolutional VAE -- compress each 64x64x3 frame to a latent z ------
class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_size=32):   # 32 for racing, 64 for doom
        super().__init__()
        self.c1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 128, 4, stride=2)
        self.c4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu     = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsig = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_dec = nn.Linear(latent_size, 1024)
        self.d1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.d2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.d3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.d4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def encode(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h))
        h = F.relu(self.c3(h)); h = F.relu(self.c4(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logsig(h)           # mu, log-sigma

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).unsqueeze(-1).unsqueeze(-1)
        h = F.relu(self.d1(h)); h = F.relu(self.d2(h)); h = F.relu(self.d3(h))
        return torch.sigmoid(self.d4(h))

    def forward(self, x):
        mu, logsig = self.encode(x)
        sigma = logsig.exp()
        z = mu + sigma * torch.randn_like(sigma)          # reparameterized sample
        return self.decode(z), mu, logsig

def vae_loss(recon, x, mu, logsig):
    # reconstruction + KL toward N(0, I): KL bounds capacity & keeps z well-behaved
    rec = F.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + 2 * logsig - mu.pow(2) - (2 * logsig).exp())
    return rec + kld

# ---- M: MDN-RNN -- distribution over the NEXT latent, plus reward & done -----
class MDNRNN(nn.Module):
    def __init__(self, latents=32, actions=3, hiddens=256, gaussians=5):
        super().__init__()
        self.latents, self.gaussians = latents, gaussians
        self.rnn = nn.LSTM(latents + actions, hiddens)
        # mixture: K means + K stds + K weights, over `latents` dims; + reward + done
        self.head = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, actions, latents):
        # actions:(T,B,A)  latents:(T,B,L)  -> mixture params over next latent
        T, B = actions.size(0), actions.size(1)
        out, _ = self.rnn(torch.cat([actions, latents], dim=-1))
        o = self.head(out)
        stride = self.gaussians * self.latents
        mu    = o[:, :, :stride].view(T, B, self.gaussians, self.latents)
        sigma = o[:, :, stride:2 * stride].view(T, B, self.gaussians, self.latents).exp()
        logpi = F.log_softmax(o[:, :, 2 * stride:2 * stride + self.gaussians], dim=-1)
        reward = o[:, :, -2]
        done   = o[:, :, -1]                              # logit; threshold at 0.5
        return mu, sigma, logpi, reward, done

def gmm_loss(target, mu, sigma, logpi):
    # negative log-likelihood of `target` (next z) under the Gaussian mixture,
    # done with log-sum-exp for numerical stability
    target = target.unsqueeze(-2)                         # broadcast over mixtures
    log_g = Normal(mu, sigma).log_prob(target).sum(-1)    # per-component log N
    log_g = logpi + log_g
    m = log_g.max(dim=-1, keepdim=True)[0]
    log_prob = m.squeeze(-1) + torch.log(torch.exp(log_g - m).sum(-1))
    return -log_prob.mean()

# ---- C: a single linear layer over [z_t ; h_t] -- the only reward-trained part
class Controller(nn.Module):
    def __init__(self, latents, hiddens, actions):
        super().__init__()
        self.fc = nn.Linear(latents + hiddens, actions)   # W_c, b_c (a few hundred params)
    def action(self, z, h):
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))   # bound the action

# ---- the agent loop: V sees the present, M's hidden state carries the future -
def rollout(controller, env, vae, rnn_cell):
    obs = env.reset(); h, c = rnn_cell.initial_state()
    total = 0.0; done = False
    while not done:
        z, _ = vae.encode(preprocess(obs))                # present
        a = controller.action(z, h)                       # acts on present + predicted future
        obs, reward, done, _ = env.step(a)
        total += reward
        _, _, _, _, _, (h, c) = rnn_cell(a, z, (h, c))    # advance M's memory
    return total

# ---- optimize C with CMA-ES: only the scalar return matters, fully parallel --
def train_controller(env_factory, vae, rnn, popsize=64, rollouts_per=16):
    es = CMAES(num_params=controller_param_count(), popsize=popsize)
    while not solved:
        candidates = es.ask()                             # population of param vectors
        fitness = []
        for params in candidates:                         # each independent -> parallel
            c = controller_from(params)
            returns = [rollout(c, env_factory(), vae, rnn) for _ in range(rollouts_per)]
            fitness.append(sum(returns) / len(returns))   # average return = fitness
        es.tell(candidates, fitness)
    return es.best()

# To learn INSIDE the dream, swap env_factory for a wrapper whose step() samples
# z_{t+1} ~ mixture(M) at temperature tau, returns predicted reward, and ends the
# episode when the done logit exceeds 0.5 -- no frames, no V, pure latent rollouts.
```

So the causal chain, start to finish: a large model would help, but training one against reward founders on credit assignment — so move the capacity into a world model trained by dense self-supervised prediction (no credit-assignment problem), and leave only a tiny controller to face reward. V is a VAE because I'll sample in its latent and need it smooth and capacity-limited; M is an MDN-RNN because the dynamics are multimodal and a point prediction would blur distinct futures; C is a single linear layer over $[z_t;h_t]$ so it sees the present plus M's compressed view of the future and can act reflexively, and it's small enough that CMA-ES can optimize it with only rollout returns. Because M *is* a latent transition-plus-done model, it can stand in for the environment and C can be trained entirely inside the dream — except a controller will cheat an approximate model, so M's stochasticity, dialed by temperature, makes the dream un-exploitable and forces a policy that transfers back to reality; and when one pass of random data isn't enough, an iterative collect–retrain loop, with M's own prediction error doubling as a curiosity reward, closes the gap.
