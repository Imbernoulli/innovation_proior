ICM's numbers came back and they point straight at its successor. Curiosity did exactly what I hoped on two games: Tutankham went from PPO's two dead seeds to all three at ~106–113 (mean 109, every seed nonzero) — the variance collapsed, exploration stopped being a coin flip. Frostbite's mean jumped to 1156, but look at the per-seed eval, {232, **3066**, 170}: one seed found a deep policy and the other two landed near where PPO already was. That's a jackpot, not a dependable lift. And on Private Eye — the game I flagged — ICM scored a flat 0.0 on *every* seed, `nonzero_rate` 0.0, never one reward. Note one good thing: the Private Eye `auc` is now ~−29 and *bounded*, where PPO had a −535 seed; curiosity stopped the destructive wandering into penalties. So curiosity made the agent explore its controllable surroundings rather than blunder — but it could not cross the gap.

The reason is the one I suspected when I built it: the forward-prediction error is large only while the controllable dynamics near the agent are unmastered, and it *decays toward zero* the moment the forward model has learned those local transitions. On Private Eye the local dynamics at the start are quickly learnable, so the bonus on that region collapses before the agent has crossed the long reward-free gap — the drive runs out at the first mastered region. So two things are wrong and they share a root. First, the novelty signal decays too fast and too *locally* — it tracks "have I learned the dynamics right here," when a long-horizon game needs "is this state still unfamiliar *overall*." Second, ICM does a lot of work to get there: an encoder, an inverse model, and a forward model, three coupled networks whose interaction is exactly what makes its Frostbite result swing from 170 to 3066 across seeds. If I want a *global* and a *more stable* novelty detector, I should look for a signal that is about global familiarity rather than local predictability, and far simpler, so there's less to go unstable.

Stay with prediction error — it's cheap, a single forward pass, and the fixed loop runs hundreds of parallel envs, so cheap-and-scalable still matters. But be precise about *why* ICM's forward model decays locally and gets stuck, because the diagnosis points at the fix. A prediction error can come from four things: (1) too little training data near this input — *epistemic* uncertainty, exactly the novelty I want; (2) the target being predicted is genuinely *stochastic* — *aleatoric* uncertainty, the noisy-TV source; (3) the model lacking the inputs or capacity to represent the target; (4) the optimizer failing to fit a representable target. ICM spends its inverse-dynamics machinery beating down (2) — and it works, which is why Tutankham stabilized — but its *forward* error still decays to zero as soon as (1) is locally resolved, because once the model has the local dynamics there's nothing left to predict wrong. I want a bonus driven by (1) and *only* (1), and one whose (1) is about global familiarity, not whether one transition's dynamics are locally solved.

So instead of fighting the stochasticity and locality of a *forward* problem, what if I choose a prediction problem whose answer is *deterministic* and *inside the predictor's own model class*? Then (2) and (3) are gone by construction and the only thing that can keep the error high is (1): too little data near this state. What deterministic function of the observation has nothing to do with dynamics and everything to do with "have I seen states like this"? Take a second neural network, initialize it randomly, and *freeze* it — call it the target $f:\mathcal{O}\to\mathbb{R}^k$, a fixed arbitrary embedding. Train a *predictor* $\hat f$ by gradient descent on the agent's observations to mimic it,
$$\min_{\theta_{\hat f}}\;\mathbb{E}_x\big\|\hat f(x;\theta_{\hat f})-f(x)\big\|^2,$$
and let the leftover error be the bonus,
$$i_t=\big\|\hat f(s_{t+1})-f(s_{t+1})\big\|^2.$$
On observations the predictor has trained on (and ones near them) gradient descent has pulled $\hat f$ onto $f$, so the error is small; on *globally novel* observations it hasn't, so the error is high. The error *is* a global-novelty signal, driven only by the amount of relevant data seen — no inverse model, no forward model, no encoder to co-train. That answers both of ICM's problems at once: it's global and slowly-decaying rather than local-and-quick-to-die, and it's dramatically simpler, so there's far less to make it seed-fragile the way ICM's three-network coupling was.

I have to confront the obvious objection: couldn't a powerful predictor just mimic $f$ *everywhere*, including unseen states, collapsing the error globally? In the limit a perfect mimic exists. The question is whether SGD *overgeneralizes* like that, and it doesn't: a predictor only becomes accurate on a region once it has seen enough examples *from that region*, so the error stays high on the genuinely unseen — precisely the behavior a novelty detector needs. There's also a clean reading of *what* the error estimates, which tells me to make the predictor deeper than the target. Osband's randomized-prior view: distilling a fixed random function is one ensemble member fit to the constant-zero target, and the predictor-target MSE estimates the *predictive variance* — an uncertainty estimate. So I give the predictor extra trainable layers beyond the target's structure, so it can fit the target on visited data while still failing on the unseen.

Now feeding this to the fixed PPO loop, there's an episode subtlety to settle before just adding $i_t$ to $e_t$. If the intrinsic return is *episodic* (truncated at game over), a game over zeroes all future intrinsic return, so the agent becomes *risk-averse* about the dangerous-but-novel maneuvers I most want — and Private Eye is full of those. The real cost of a game over is only replaying the (now-boring) start, so the intrinsic return should be *non-episodic*. But the *extrinsic* return must stay episodic, or an agent could grab an early reward, suicide, and farm it in a loop. Two streams, different episodicity. Returns are linear in rewards, $R=R_E+R_I$, so I fit *two value heads* $V_E,V_I$, give each its own discount — $\gamma_E=0.999$ because extrinsic rewards are far apart, $\gamma_I=0.99$ because the bonus is transient and stepping-stone-like — compute advantages per stream and sum them, $A=A_I+A_E$, and run the fixed PPO update on the combined advantage. Separate heads also fit a *stationary* extrinsic reward and a *non-stationary* intrinsic one (the bonus decays as the predictor learns) without forcing one head to track both.

Two normalizations make or break it, and the second is specific to a *frozen random* target. First, the intrinsic reward's raw scale drifts (it's an MSE of arbitrary embeddings that shrinks as learning proceeds), so divide $i_t$ by a running std of the intrinsic *returns* to keep the bonus on a consistent scale. Second, the target is frozen at random init, so unlike a trained net it *cannot adapt to the input scale*; if observations arrive at an arbitrary scale the random target squashes them into a near-constant output and the error becomes meaningless. So whiten the *observations* into predictor and target — subtract a running mean, divide by a running std, clip to $[-5,5]$ — initialized from a short random rollout before training. This whitening goes to predictor/target only, not the policy net.

```python
import torch
import torch.nn as nn

K = 512  # embedding dimension of target/predictor

class RND(nn.Module):
    # target: frozen random net (deterministic target, in predictor's model class).
    # predictor: trained to distill the target on visited observations; DEEPER, so it fits the
    # target where it has data but cannot trivially copy it on the unseen.
    def __init__(self):
        super().__init__()
        def conv():
            return nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4), nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(), nn.Flatten())
        self.target = nn.Sequential(conv(), nn.Linear(3136, K))                 # frozen
        self.predictor = nn.Sequential(conv(), nn.Linear(3136, K), nn.ReLU(),
                                       nn.Linear(K, K), nn.ReLU(), nn.Linear(K, K))  # deeper
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):                          # obs: single normalized+clipped frame
        return self.predictor(obs), self.target(obs)

    def intrinsic_reward(self, next_obs):
        with torch.no_grad():
            pred, tgt = self.forward(next_obs)
            return (pred - tgt).pow(2).sum(dim=1)    # i_t = ||f_hat - f||^2 : global novelty

    def distillation_loss(self, obs):
        pred, tgt = self.forward(obs)
        return (pred - tgt.detach()).pow(2).sum(dim=1).mean()

# loop: init obs-norm from a random rollout; per rollout compute i_t, normalize by intrinsic-return std,
#   R_I/A_I NON-EPISODIC gamma_I=0.99 (head V_I), R_E/A_E EPISODIC gamma_E=0.999 (head V_E),
#   A = A_I + 2*A_E, then PPO clip-loss with V = V_E + V_I and distillation_loss on ~25% of the batch.
```

So the delta from step 2: swap the inverse/forward curiosity module for a frozen-random-target + trained-predictor pair, emit the distillation error as $i_t$, normalize by a running intrinsic-return std, whiten the observations into the target/predictor, and value it with the two-stream non-episodic/episodic heads. Reading ICM's shape, here's what I expect. The global, slowly-decaying signal should finally give the agent a reason to *keep* moving past the first mastered region — so Private Eye is the game I'm watching: if a non-decaying global bonus gets even one seed to cross the gap and register a real Private Eye return, that's the breakthrough ICM couldn't reach. Tutankham and Frostbite I expect to hold roughly where curiosity put them, perhaps trading ICM's high-variance Frostbite peak (the 3066 seed) for more stability, since RND has far less coupled machinery to swing seed-to-seed.

The causal chain in one breath: ICM stabilized Tutankham but its forward error *decays locally* and died before crossing Private Eye's gap (flat 0 on all seeds), and its three coupled nets made Frostbite a one-seed jackpot → I want a *global*, slowly-decaying, *simpler* novelty signal → so pick a prediction problem with a *deterministic, in-class* answer: distill a frozen random target, $i_t=\|\hat f(s_{t+1})-f(s_{t+1})\|^2$, whose error is driven only by lack of nearby data (an uncertainty estimate, hence the deeper predictor) → feed it to the fixed PPO loop with split streams — intrinsic non-episodic so the agent isn't risk-averse about dangerous-but-novel maneuvers, extrinsic episodic so it can't suicide-farm — two heads $V=V_E+V_I$, $\gamma_E=0.999>\gamma_I=0.99$ → normalize the bonus and whiten the observations into the frozen target so its embedding carries information → watching Private Eye for the cross-the-gap result ICM couldn't get.
