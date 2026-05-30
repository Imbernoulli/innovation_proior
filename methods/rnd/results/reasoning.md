The thing I want is an exploration bonus for the genuinely hard sparse-reward games — Montezuma's Revenge, where positive rewards are hundreds of steps apart and a plain PPO agent on the extrinsic reward almost never sees a single one. The bonus has to make the agent seek out *novel* states, $r_t=e_t+i_t$ with $i_t$ large where the agent hasn't been. And because the regime that actually works in modern RL is massive parallelism — hundreds of environments, billions of frames — the bonus has to be *cheap* and *scalable*. That immediately makes me nervous about the count-style and density-model bonuses: pseudo-counts need a learned density model over images, and prediction-improvement / learning-progress bonuses need to track how much a model got better per datapoint — both are heavy and fiddly to run across thousands of actors. So I'd like something I can compute with a single forward pass of a network on a batch.

The cheap idea is prediction error: train a model on the agent's experience, and where its error is high, the state is novel — because neural networks have lower error on inputs similar to what they've trained on. The obvious instantiation is a forward dynamics model: predict the next observation (or its features) from the current observation and action, and reward the agent by the prediction error. Let me think about whether that actually behaves, by walking the agent into the worst case. Put a television showing white-noise static in the room. The next-frame content of the static is, by construction, random — no model can predict it. So the forward-prediction error on those transitions is *irreducible*: it never decays no matter how long the agent stares. The bonus stays high forever, and a reward-maximizing agent will simply park itself in front of the TV, or by a coin flip, or — and this is the version that actually bites in Atari — at a transition whose outcome is randomized by sticky actions, like a room boundary where one extra step might or might not cross. The agent oscillates there, farming the irreducible error, and stops exploring. That's the noisy-TV problem, and it's fatal for naive prediction-error curiosity.

Let me be precise about *why* the forward model fails, because the diagnosis points at the fix. A prediction error can come from four things: (1) too little training data near this input — the predictor hasn't seen states like this, which is *epistemic* uncertainty and is exactly the novelty signal I want; (2) the target being predicted is genuinely *stochastic* — *aleatoric* uncertainty, which is the noisy-TV source; (3) the model being misspecified — it lacks the inputs or the capacity to represent the target; (4) the optimizer failing to fit a target the model class actually contains. I want a bonus driven by (1) alone. The forward model gets clobbered by (2): the next observation is a stochastic function of the current one whenever the environment has any randomness, so factor (2) is baked into the very prediction problem. Prediction-improvement methods try to subtract off (2) and (3) by measuring learning progress instead of raw error, but that's the expensive route I'm trying to avoid.

So instead of fighting the stochasticity of the forward problem, what if I *choose a prediction problem whose answer is deterministic*? If the target I'm predicting is a fixed deterministic function of the current observation, then factor (2) is gone by construction — there's no randomness in the answer. And if I also make sure the target is *inside the predictor's own model class* — something the predictor can in principle represent exactly — then factor (3) is gone too. What deterministic function of the observation can I conjure that has nothing to do with the (irrelevant) dynamics and everything to do with "have I seen states like this"? Take a second neural network, initialize it randomly, and *freeze it*. Call it the target $f:\mathcal{O}\to\mathbb{R}^k$. Its output on an observation is a fixed, deterministic, arbitrary embedding. Now train a *predictor* network $\hat f:\mathcal{O}\to\mathbb{R}^k$, by gradient descent on the agent's collected observations, to mimic the target:
$$\min_{\theta_{\hat f}}\;\mathbb{E}_x\big\|\hat f(x;\theta_{\hat f})-f(x)\big\|^2.$$
This is *distilling* a randomly initialized network into a trained one. The intrinsic reward is just the leftover error on the new observation,
$$i_t=\big\|\hat f(s_{t+1})-f(s_{t+1})\big\|^2.$$
Check it against the four factors. The target $f$ is deterministic, so no aleatoric (2). The target is a neural network and I can make the predictor at least as expressive, so $f$ lives inside the predictor's model class — no misspecification (3). What's left to drive the error is (1): on observations the predictor has trained on (and ones similar to them) the distillation error is small, because gradient descent has pulled $\hat f$ toward $f$ there; on *novel* observations the predictor hasn't been pulled toward $f$ yet, so the error is high. The error *is* a novelty signal, purely from the amount of relevant training data. And it's exactly as cheap as I wanted: one forward pass each of two small networks.

Now I have to confront the obvious objection: if the predictor is powerful enough, couldn't gradient descent just make it mimic $f$ *everywhere*, including on states it's never seen — collapsing the error to zero globally? In the limit $f$ itself is a perfect predictor of $f$, so a perfect mimic exists. If the predictor found it, the bonus would die everywhere and exploration would stop. The question is empirical — does SGD *overgeneralize* like that? I can settle it with a toy on MNIST: train a predictor to match a random target on a dataset that is mostly zeros plus a few examples of some target class, vary the proportion, and measure test error on held-out target-class images. The zeros stand in for states seen many times; the target class for states seen rarely. If the predictor overgeneralized, the test error on the rare class would be low regardless. What I'd expect — and the reason I trust this works — is that the test error *decreases as the number of target-class training examples increases*: the predictor only becomes accurate on a region once it has seen enough examples *from that region*. So SGD does not overgeneralize in the harmful way; the error stays high on the genuinely unseen, which is precisely the behavior a novelty detector needs.

There's a clean way to see *what* this error is estimating, which also tells me the predictor should be a bit *deeper* than the target. Consider Osband's randomized-prior trick for uncertainty: an ensemble $g_\theta=f_\theta+f_{\theta^*}$ with $\theta^*$ drawn from a prior, each member fit by minimizing $\mathbb{E}\|f_\theta(x)+f_{\theta^*}(x)-y\|^2$ plus a prior regularizer; the spread of the ensemble approximates the posterior, hence predictive uncertainty. Specialize the regression targets to $y=0$: then minimizing $\mathbb{E}\|f_\theta(x)+f_{\theta^*}(x)\|^2$ is exactly distilling the random function $f_{\theta^*}$ (the predictor $f_\theta$ learns to cancel it where it has data). Each output coordinate of my predictor/target pair is then like one ensemble member with shared parameters, and the MSE between predictor and target is an estimate of the *predictive variance* — the uncertainty in predicting the constant-zero function. So RND's distillation error is a cheap uncertainty estimate, and I want the predictor to have *extra* trainable layers beyond the target's structure so it can actually fit the target on visited data (rather than being a frozen mirror), while still being unable to fit it on data it hasn't seen.

Now I need to feed this bonus into a policy optimizer, and there's a subtlety about *episodes* I should think through before just adding $i_t$ to $e_t$. Imagine the agent attempting a risky maneuver to reach a suspected secret room — high chance of a game over, but high curiosity payoff if it succeeds. If I treat the intrinsic return as *episodic* (truncated at game over), then a game over zeroes out all future intrinsic return, so the agent learns to be *risk-averse* about exactly the dangerous-but-novel maneuvers I want it to attempt. But the real cost of a game over isn't "all future novelty is gone forever" — it's just the opportunity cost of replaying the (now-boring) start of the game. So the intrinsic return should be *non-episodic*: it should reflect all the novel states the agent could find in the future, regardless of episode boundaries. (Episodic intrinsic reward also leaks task information through the episode structure.) But I can't make the *extrinsic* return non-episodic — that would be exploitable by an agent that grabs an early reward, deliberately gets a game over, and farms that reward in an endless cycle. So I want the intrinsic stream non-episodic and the extrinsic stream episodic — two streams with genuinely different structure.

How do I value two streams with different episodicity (and, I'll want, different discounts)? The return is *linear* in the rewards, so it decomposes as $R=R_E+R_I$. That means I can fit *two separate value heads*, $V_E$ trained on the episodic extrinsic returns and $V_I$ trained on the non-episodic intrinsic returns, and recombine $V=V_E+V_I$ when I need a value. The same linearity lets each stream carry its own discount: the extrinsic reward needs a long horizon because rewards are far apart, so $\gamma_E=0.999$; the intrinsic reward works better at $\gamma_I=0.99$ (a higher intrinsic discount empirically hurts — the bonus is transient and stepping-stone-like, so a shorter horizon for it is fine). I compute advantages per stream and add them, $A=A_I+A_E$, and run PPO on the combined advantage. Two heads also just give the value function more supervisory signal, and there's a structural reason they help here: the extrinsic reward function is *stationary* while the intrinsic one is *non-stationary* (the bonus decays as the predictor learns), so it's cleaner to fit them separately than to force one head to track a moving target plus a fixed one.

Two normalization issues will make or break this in practice, and the second is specific to using a *random frozen* target. First, the intrinsic reward's scale varies wildly across games and across time (it's an MSE of arbitrary embeddings that shrinks as learning proceeds), so a fixed bonus coefficient won't transfer. Normalize $i_t$ by dividing by a running estimate of the standard deviation of the intrinsic *returns*, keeping the bonus on a consistent scale. Second — and this is the one that's easy to miss — the target network's parameters are *frozen at random init*, so unlike a normal trained network it *cannot adapt to the scale of the input observations*. If the observations come in at some arbitrary scale, the random target can squash them into a tiny output range and the embedding ends up nearly constant, carrying almost no information about the input, and then the distillation error is meaningless. So I have to normalize the *observations* feeding the predictor and target: whiten each dimension by subtracting a running mean and dividing by a running std, then clip to $[-5,5]$. I initialize these normalization statistics by stepping a *random* agent in the environment for a few steps before training starts, so the stats are sane from the first update. This observation normalization is applied to the predictor and target but *not* to the policy network (which has its own simple $x/255$ scaling and a 4-frame stack); the predictor/target see a single normalized frame.

Putting the loop together: initialize the observation-normalization stats with a random rollout; then repeatedly collect rollouts under the current PPO policy, computing $i_t=\|\hat f(s_{t+1})-f(s_{t+1})\|^2$ for each step and updating the reward-normalization stats; per rollout, normalize the intrinsic rewards, compute non-episodic intrinsic returns/advantages and episodic extrinsic returns/advantages, sum the advantages, update the observation stats; then for a few epochs jointly optimize the policy by the PPO loss and the predictor by the distillation loss. The bonus is optimizer-agnostic — I use PPO because it's robust and needs little tuning. Now the code.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 512  # embedding dimension of target/predictor

class RND(nn.Module):
    # target: frozen random net (deterministic target, in predictor's model class).
    # predictor: trained to distill the target on visited observations; DEEPER so it can fit
    # the target on seen data but does not trivially copy it everywhere.
    def __init__(self):
        super().__init__()
        def conv():
            return nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4), nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(), nn.Flatten())
        self.target = nn.Sequential(conv(), nn.Linear(3136, K))                 # single dense, frozen
        self.predictor = nn.Sequential(conv(), nn.Linear(3136, K), nn.ReLU(),
                                       nn.Linear(K, K), nn.ReLU(), nn.Linear(K, K))  # deeper, trainable
        for p in self.target.parameters():
            p.requires_grad = False                                              # target is frozen

    def forward(self, obs):                          # obs: single normalized+clipped frame
        return self.predictor(obs), self.target(obs)

    def intrinsic_reward(self, next_obs):
        with torch.no_grad():
            pred, tgt = self.forward(next_obs)
            return (pred - tgt).pow(2).sum(dim=1)    # i_t = ||f_hat - f||^2  (per state)

    def distillation_loss(self, obs):
        pred, tgt = self.forward(obs)
        return (pred - tgt.detach()).pow(2).sum(dim=1).mean()   # train predictor toward frozen target

class RunningNorm:
    # running mean/std for observation whitening and for intrinsic-return std.
    def __init__(self, shape):
        self.mean = torch.zeros(shape); self.var = torch.ones(shape); self.count = 1e-4
    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean; tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

def normalize_obs(x, stats):                         # predictor/target input only (NOT policy)
    return ((x - stats.mean) / (stats.var.sqrt() + 1e-8)).clamp(-5.0, 5.0)

# --- training loop sketch (PPO + RND), two value heads, two reward streams ---
# init: step a RANDOM agent for M steps, update obs-norm stats.
# per rollout of length 128 over 128 envs:
#   for each step: a_t ~ pi; observe s_{t+1}, e_t (clipped [-1,1])
#                  i_t = rnd.intrinsic_reward(normalize_obs(s_{t+1}, obs_stats))
#                  update intrinsic-return-std stats with i_t
#   normalize intrinsic rewards:  i_t /= ret_std          # consistent bonus scale
#   R_I, A_I  from i_t  NON-EPISODIC, gamma_I = 0.99       # intrinsic value head V_I
#   R_E, A_E  from e_t  EPISODIC,     gamma_E = 0.999      # extrinsic value head V_E
#   A = c_I * A_I + c_E * A_E                              # c_I = 1, c_E = 2
#   update obs-norm stats with the rollout
#   for epoch in range(4), minibatch in range(4):
#       optimize theta_pi by PPO clip-loss on (A, R_E, R_I) with V = V_E + V_I, Adam lr 1e-4
#       optimize theta_predictor by distillation_loss on 25% of the batch (keep prob 0.25)
```

The causal chain in one breath: sparse-reward games need a cheap, scalable, directed-exploration bonus → prediction error is cheap but a forward-dynamics bonus is captured forever by irreducible *aleatoric* error (the noisy-TV trap) → so pick a prediction problem with a *deterministic* answer *inside the predictor's model class*: distill a fixed randomly-initialized target network, $i_t=\|\hat f(s_{t+1})-f(s_{t+1})\|^2$, whose error is driven only by lack of nearby training data — i.e. novelty (and is an uncertainty estimate, à la randomized priors, which is why the predictor is made deeper than the target) → feed it to PPO, but split the streams: the intrinsic return non-episodic (so the agent isn't risk-averse about dangerous-but-novel maneuvers) and the extrinsic return episodic (so it can't suicide-farm rewards), valued by *two heads* $V=V_E+V_I$ with separate discounts $\gamma_E=0.999>\gamma_I=0.99$ and summed advantages → normalize the intrinsic reward by a running return-std, and crucially whiten+clip the *observations* into the frozen random target so its embedding carries information at all.
