Let me start from what actually breaks when I try to learn a policy from a fixed dataset and never touch the environment. I have `D`, a pile of transitions `(s, a, r, s')` from some behavior policy I don't get to see, and I want the best continuous-control policy `pi(s)` I can squeeze out of it. The temptation is to just run my favorite off-policy actor-critic on `D` as if the replay buffer happened to be frozen. I know exactly where that dies. To train the critic I regress `Q(s,a)` onto a bootstrapped target `r + gamma·Q(s', a')`, and `a'` is whatever my current policy proposes at `s'`. Online that's fine — if `a'` is a bad idea I'll go execute it, get a low reward, and the critic gets corrected. Offline there is no execution and no correction: `a'` is very often an action the dataset simply never contains at `s'`, and a neural net asked for `Q` at an out-of-distribution input gives me an essentially arbitrary extrapolation. Worse than arbitrary — biased upward, because the actor is trained to *maximize* `Q`, so it actively seeks out exactly the OOD actions the critic happens to overrate. That overrated value goes into the next target, the next backup inherits it, and the whole value surface inflates and detaches from reality. The policy chases the inflation off into the part of action space I have no data for, and the run collapses. So the real problem isn't "do RL on a buffer", it's: how do I stop the critic from trusting out-of-distribution actions, while still letting the policy exploit the in-distribution actions that are genuinely good?

I should be honest about the second pressure too, because it shapes what kind of answer I'll accept. The field around me has piled up offline methods of increasing complexity — conservative critic penalties, big critic ensembles, learned generative behavior models, expectile value functions — and almost every one of them ships its headline idea bundled with a fistful of "minor" choices: a different network depth, a bigger batch, some normalization between layers, a tweaked learning rate, an actor pre-training phase. Those choices are basically never ablated against a clean baseline, so when a paper claims +N points I genuinely cannot tell whether it's the algorithm or the bundle. That bothers me enough that I want my answer to be *minimal and attributable*: each piece individually justified, each piece individually removable so I can see what it bought, no secondary networks, no real compute overhead, and no extra parameters — the cap on parameter count means whatever I do has to be in the *loss and the target construction*, not in capacity.

Let me lay out the tools I actually have. The strongest online base I trust in continuous control is TD3, and it's worth being precise about *why* each of its pieces exists, because they're all aimed at the same overestimation disease I'm fighting, just in the online form. The root fact is Thrun and Schwartz's: a greedy target `max_{a'} Q(s',a')` is upward-biased under noise, because for zero-mean error `eps`, `E[max_{a'}(Q+eps)] >= max_{a'} Q` — the max picks out whichever action got the luckiest positive error. Fujimoto showed the same thing happens for a deterministic actor-critic even though the max is implicit in the policy gradient. TD3's three answers: keep two critics and bootstrap from their minimum, `y = r + gamma·min_{i=1,2} Q_{theta'_i}(s', a')`, so the min is an approximate upper-bound suppressor — it biases me toward *under*estimation, and underestimation is self-correcting in a way overestimation isn't, because the policy just avoids the actions it underrates instead of chasing them. Add clipped noise to the target action, `a' = pi_{phi'}(s') + clip(N(0,sigma),-c,c)` with `sigma=0.2, c=0.5`, so the target can't overfit a razor-thin spike in `Q` — nearby actions are forced to share value. And update the actor and the slow targets only every `d=2` critic steps, so the critic error settles before each policy move. These are good and I'm keeping them. But notice what none of them do: not one keeps the policy near the dataset. Offline, the actor walks straight off the edge of the data and the clipped-min can't save a critic whose entire off-distribution surface is unconstrained garbage — `min` of two garbage numbers is still garbage. So TD3 alone is necessary machinery but not the offline fix.

What's the cheapest thing that *is* an offline fix? The minimalist move I keep coming back to is: just add a behavior-cloning term to the actor so it's penalized for straying from the dataset action. Concretely the deterministic-policy-gradient actor update becomes `pi = argmax_pi E_{(s,a)~D}[ Q(s,pi(s)) - (pi(s)-a)^2 ]` — maximize value, but pay a squared penalty for deviating from the logged action `a` at that state. One line on top of TD3. I like it because it's the opposite of bundling complexity: no generative model, no extra network, no sampled-action logsumexp. There's one real subtlety I have to get right, though, and it's about scale. The BC term `(pi(s)-a)^2` is bounded — actions live in `[-1,1]^m`, so it's at most about 4 — but `Q` scales with the reward magnitude, which is totally arbitrary across tasks. If I just write `Q - (pi-a)^2`, then on a high-reward task `Q` dwarfs the BC term and the policy ignores the data, and on a low-reward task the BC term dominates and I get pure imitation. So I need to normalize the value term to put it on a comparable footing. The clean way is to divide `Q` by the average magnitude of `Q` itself: `lambda = alpha / mean(|Q(s,a)|)`, and use `lambda·Q - (pi-a)^2`. Crucially `lambda` is a *stop-gradient* scalar — it rescales the loss, it does not change the direction of the `Q` gradient, because I'm differentiating `pi`, not the normalizer. With `alpha=2.5` this is TD3+BC, and it genuinely matches much heavier methods. So that's my floor.

Now I want to find where TD3+BC is leaving value on the table, because I don't think one BC term on the actor is the end. Stare at the critic target again. I regularized the *actor* at the training states `s` — fine, at `s` the policy is pulled toward `a`. But the critic's target is `r + gamma·min Q(s', a')` with `a' = pi_target(s') + noise`. That next action `a'` is generated at the *next* state `s'`, and nothing in the actor's BC term at `s` guarantees `pi(s')` is in-distribution at `s'`. So the bootstrap can still pick an OOD `a'`, the critic can still overrate it, and I've reintroduced the exact loop I was trying to kill — just one bootstrap step removed from where I patched it. The actor penalty fixes "what action do I take at states I've seen"; it does nothing about "what action does the target *assume I'll take next*". Those are different leaks. The actor BC plugs one; the target is still open.

So I want a penalty *inside the critic target* too. And here the general framing is already sitting in the literature: behavior-regularized actor-critic says there are exactly two places to inject a divergence `D(pi(·|s), pi_b(·|s))` from the behavior policy. You can put it in the actor objective — a *policy regularization*, which is what the BC term I already have is — or you can put it in the critic target — a *value penalty*, subtracting `alpha·D` from the bootstrapped value so the backup is pessimistic exactly where the policy departs from behavior:
`min_Q E[ (r + gamma·(Q̄(s',a') - alpha·D(pi(·|s'),pi_b(·|s'))) - Q(s,a))^2 ]`.
That's the missing half. The actor penalty keeps the *acting* policy near data; the value penalty keeps the *bootstrap* from trusting a next-action that's drifted off data. They're complementary, and TD3+BC simply never took the value-penalty half.

But the general framework was written with stochastic policies and density divergences — KL, MMD, Wasserstein — and they tried all three and found no consistent winner, and every one of them needs a separately learned behavior model `pi_b` to take a sample-based divergence against. That's exactly the bundled complexity I'm trying to avoid: a whole extra generative network and its hyperparameters, to compute a divergence whose particular form doesn't even seem to matter. So let me ask what divergence I'd actually use here. My policy is *deterministic* — it's TD3, `pi(s)` is a point, not a distribution. KL between a point mass and anything is degenerate; MMD and Wasserstein against a learned `pi_b` are overkill. The natural "divergence" between the policy's action and the dataset's action at a state is just the squared Euclidean distance between them — which is precisely the BC term I already used on the actor. So I don't need a behavior model at all: in the critic target the divergence is `(a' - â')^2`, where `â'` is the *dataset's own* recorded next action at `s'`. The dataset already hands me `(s, a, r, s', â')` if I keep the next action around — `â'` is just the action logged at the next transition. So the value penalty becomes: take the bootstrapped `min Q(s', a')`, and subtract a squared penalty for how far the policy's next action `a'` is from the dataset's next action `â'`. No extra network, one subtraction.

Let me write the target out properly. Next action with target smoothing, `a' = clip(pi_target(s') + clip(N(0,sigma),-c,c), -1, 1)`. Bootstrapped value `q = min_{i=1,2} Q_{theta'_i}(s', a')`. Value penalty `q <- q - beta_critic·(a' - â')^2`, summed over action dims. Then `y = r + gamma·(1-done)·q`. And the actor keeps its own penalty, `lambda·Q(s,pi(s)) - beta_actor·(pi(s)-a)^2`. Now I'm using a BRAC value penalty *and* a BRAC policy regularization at once, instantiated for a deterministic policy with the cheap squared distance.

Here's the thing I want to push on, the thing the framework left coupled: should `beta_actor` and `beta_critic` be the *same* number? The original framework gave both sides one shared `alpha`. But think about what each side is actually doing. The actor penalty controls how conservative the policy is *when it acts* — how willing it is to deviate from the logged action to chase value. The critic penalty controls how distrustful the *bootstrap* is of off-distribution next-actions. Those are genuinely different jobs with different right answers per environment. On a task where the dataset is broad and the dynamics forgiving, I might want a tiny actor penalty (let the policy improve freely) but a meatier critic penalty (the bootstrap still needs guarding). On a narrow dataset the reverse. Forcing one coefficient to serve both means every task gets a single point on a one-dimensional trade-off, when the real trade-off is two-dimensional. So decouple them: `beta_1` for the actor, `beta_2` for the critic, tuned separately. This is cheap — it's two scalars instead of one — and it's the generalization of the framework that the deterministic-MSE instantiation makes natural. I'd expect, and I'd want to check, that the actor penalty is the load-bearing one (it's the thing standing between me and the OOD collapse), and that the critic penalty helps less but completes the picture — but the point of decoupling is precisely that I don't have to assume they move together.

OK, so on the *loss* side I now have: TD3's machinery (twin critics, clipped-min, target smoothing, delayed updates, soft targets at `tau=5e-3`), plus a decoupled two-sided BC penalty with squared distance, plus the TD3+BC `lambda = 1/mean(|Q|)` normalization on the actor's value term. Let me fold the normalization in cleanly. The actor loss I'll minimize is `mean( beta_1·(pi(s)-a)^2 - lambda·Q(s,pi(s)) )` with `lambda = stopgrad(1/mean|Q(s,pi(s))|)` when I want the normalization on. Notice that once `lambda` carries the `1/mean|Q|` factor, the leftover constant `alpha` from TD3+BC has effectively been absorbed into how I tune `beta_1`: the actor's balance is set by `beta_1` against a value term that's already normalized to order one. So I don't carry a separate `alpha=2.5`; I tune `beta_1` per task and let `lambda` handle the scale. That's the actor.

Now I have to confront the architectural and optimization choices, because this is where the "minor bundle" I was suspicious of actually lives, and I refuse to copy them blindly — I want each one to either earn its place by a real argument or get dropped. Start with normalization between layers, because I have a concrete reason to believe it matters for *exactly my disease*. The whole problem is the critic extrapolating wildly on OOD actions. What does LayerNorm do to that? Suppose the last hidden representation feeding the critic's output head `w` is layer-normalized — call it `psi(s,a)`. LayerNorm rescales `psi` to (roughly) unit norm regardless of input, so `||psi(s,a)||` is a bounded constant for *any* `(s,a)`, including ones far off-distribution. Then for any input at all,
`|Q(s,a)| = |w^T relu(psi(s,a))| <= ||w||·||relu(psi(s,a))|| <= ||w||·||psi(s,a)|| <= ||w||`,
using Cauchy-Schwarz, then `||relu(x)|| <= ||x||` (ReLU only shrinks magnitudes), then the LayerNorm bound. So the critic's output is *bounded by the norm of its head weights*, even on actions it's never seen. That's not a soft preference, it's a hard cap: the OOD value literally cannot run far above the in-distribution values, because the same weight norm bounds both. That kills the runaway-extrapolation engine that feeds the overestimation loop. And the beautiful part is it does it *without* explicitly telling the policy "stay near the data" — it just makes the off-distribution surface bounded and well-behaved. So LayerNorm goes in the **critic**, between every hidden layer. Do I put it in the actor too? The actor isn't the thing extrapolating dangerous values — it's a policy bounded into `[-1,1]` by a tanh and already pulled to the data by `beta_1`. The extrapolation pathology lives in the critic's value surface, not in the policy's action surface, so normalizing the actor isn't addressing the disease; I'll leave the actor without inter-layer normalization. Asymmetric on purpose: LayerNorm in the critic, none in the actor.

Depth next. The default off-policy bases use two hidden layers, and I keep seeing later offline work quietly switch to three and call it important without much argument. The scaling intuition is straightforward — more depth gives more capacity to fit the value/policy given enough data, and offline I have a big static dataset — so three hidden layers of width 256 is the reasonable bump. I shouldn't oversell it: I'd expect it to help meaningfully on the harder tasks (the critic on long-horizon navigation needs the capacity) and to be close to a wash on the easy locomotion ones, with diminishing returns past three or four and a drop if I go absurd (six). So: three hidden layers, width fixed at 256, in both actor and critic.

Batch size and learning rate. Larger batches give lower-variance gradient estimates, which can speed convergence within a fixed one-million-step budget, and the standard heuristic is to scale the learning rate up with the batch. So on the locomotion datasets I'll push the batch to 1024 and the learning rate to `1e-3`. But I should be careful and not declare this universal, because a bigger batch isn't free — it can over-smooth and it interacts badly with some domains. My expectation is it's a clear win on Gym-MuJoCo and a loser on, say, sparse-reward navigation, where I'd keep the batch at 256 and the learning rate small. So this is a per-domain knob, not a global setting — exactly the kind of "minor choice" that I want explicit and ablatable rather than baked in.

The discount factor is the one place where the "tuning knob" is really just arithmetic, and I want to derive it rather than tune it. Default `gamma = 0.99` is fine for dense-reward locomotion. But picture a long, sparse-reward task — a maze where the only reward arrives at the goal, after up to a thousand steps. The value at the start state has to carry that terminal reward back across the whole episode, and it arrives discounted by `gamma^L`. With `gamma=0.99` and `L=1000`, `0.99^1000 ~= 4·10^-5` — the only informative signal in the entire episode is multiplied into oblivion before it reaches the start, so the critic has essentially nothing to learn from. Bump to `gamma=0.999` and `0.999^1000 ~= 0.37`: now better than a third of the terminal reward survives to the start state, and the signal can actually propagate. (Scaling the sparse reward up by a constant, say ×100, helps the magnitude but doesn't fix the discounting — `gamma^L` is the real killer, so `gamma` is the lever.) So for the long sparse-reward family I set `gamma=0.999`; for dense locomotion I leave it at `0.99`. Not a hyperparameter I sweep — a number I read off the horizon.

One inherited choice I'll deliberately *drop*: TD3+BC also normalized the state features by the dataset mean and std. I don't want it. Two reasons — I'd like this same method to be runnable online later, where a fixed dataset mean/std would go stale as the state distribution drifts, and in the offline locomotion setting the effect of state normalization is small anyway. So I trade a negligible offline gain for keeping the method simple and online-compatible. (If I did want a normalization on the features, a per-layer feature norm sits about on par with LayerNorm — but that's a side road; LayerNorm in the critic is the load-bearing one.)

Let me also nail the update *schedule*, because TD3's delayed-update structure interacts with how I've split the losses. The critic gets updated every step — it needs all the gradient steps it can get to fit a moving target. The actor and the target networks update on the delayed cadence, every `policy_freq = 2` steps. So on a delayed step I do critic-then-actor-then-soft-target; on a non-delayed step I do critic only. That's the two-timescale structure that lets the critic error shrink between policy moves, and I keep it exactly.

Now let me make sure the pieces compose into one coherent training step before I write code. Per step: sample `(s, a, r, s', â', done)`. Build the smoothed next action `a' = clip(pi_target(s') + clip(N(0,0.2),-0.5,0.5), -1, 1)`. Bootstrap `q = min(Q1_target(s',a'), Q2_target(s',a'))`, then apply the value penalty `q <- q - beta_2·sum((a' - â')^2)`. Target `y = r + gamma·(1-done)·q`. That is the core equation. If my dataset preparation also carries an empirical return-to-go, I can keep a separate calibration switch that floors `y` at that observed return; that is not the behavior-regularization idea, so it must stay explicit and optional. Critic loss `MSE(Q1(s,a), y) + MSE(Q2(s,a), y)`, step both critics. If it's a delayed step: actor action `pi = actor(s)`, value `q_pi = Q1(s, pi)`, normalizer `lambda = stopgrad(1/mean|q_pi|)`, actor loss `mean(beta_1·sum((pi-a)^2) - lambda·q_pi)`, step the actor; then soft-update all three targets at `tau`. Every signal traces to a piece I argued for: the min and smoothing and delay are TD3's overestimation/variance controls, the two `beta` penalties are the decoupled deterministic BRAC value-and-policy regularization, `lambda` is TD3+BC's scale normalization, LayerNorm in the critic is the extrapolation bound, depth/batch/`gamma` are the read-off design choices.

Let me write it as the code I'd actually ship, filling the loss and network slots in the offline actor-critic harness. The critic carries LayerNorm between hidden layers; the actor does not.

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicActor(nn.Module):
    """pi(s) = max_action * tanh(net(s)). 3 hidden layers of 256, ReLU,
    NO inter-layer normalization: the actor isn't the surface that extrapolates
    dangerous values, and a tanh already bounds its output into the action box."""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().numpy().flatten()


class Critic(nn.Module):
    """Q(s, a). 3 hidden layers of 256, ReLU, WITH LayerNorm between layers.
    LayerNorm bounds the last hidden feature norm, so |Q| <= ||w_head|| even on
    out-of-distribution actions -> caps catastrophic value extrapolation."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ReBRAC:
    """TD3 with a decoupled, deterministic behavior-regularization penalty on
    both the actor loss (beta_1) and the critic bootstrap target (beta_2),
    plus the TD3+BC value normalization on the actor."""

    def __init__(self, state_dim, action_dim, max_action,
                 actor_bc_coef=1.0, critic_bc_coef=1.0,     # beta_1, beta_2; YAML overrides per env
                 discount=0.99, tau=5e-3, lr=1e-3,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 normalize_q=True, use_mc_return_floor=False, device="cuda"):
        self.device = device
        self.beta_1 = actor_bc_coef          # actor-side BC penalty strength
        self.beta_2 = critic_bc_coef         # critic-side BC penalty strength (decoupled)
        self.discount = discount             # 0.999 on long sparse-reward tasks, else 0.99
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise     # target policy smoothing (TD3)
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq       # delayed actor/target updates (TD3)
        self.normalize_q = normalize_q
        self.use_mc_return_floor = use_mc_return_floor
        self.total_it = 0

        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def train(self, batch):
        self.total_it += 1
        # batch carries the dataset's own next action a_hat' for the critic penalty
        if self.use_mc_return_floor:
            states, actions, rewards, next_states, dones, next_actions_data, mc_returns = batch
            mc_returns = mc_returns.squeeze(-1)
        else:
            states, actions, rewards, next_states, dones, next_actions_data = batch
            mc_returns = None
        not_done = 1.0 - dones.squeeze(-1)
        rewards = rewards.squeeze(-1)

        # ---- critic update (every step) ----
        with torch.no_grad():
            # target policy smoothing: noisy, clipped next action
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action)
            # clipped double-Q bootstrap
            next_q = torch.min(self.critic_1_target(next_states, next_actions),
                               self.critic_2_target(next_states, next_actions))
            # BRAC value penalty: pessimize the bootstrap toward the dataset's
            # next action a_hat' (no behavior model needed; deterministic -> MSE)
            bc_penalty = ((next_actions - next_actions_data) ** 2).sum(-1)
            next_q = next_q - self.beta_2 * bc_penalty
            target_q = rewards + not_done * self.discount * next_q
            if mc_returns is not None:
                target_q = torch.maximum(target_q, mc_returns)

        critic_loss = (F.mse_loss(self.critic_1(states, actions), target_q)
                       + F.mse_loss(self.critic_2(states, actions), target_q))
        self.critic_1_opt.zero_grad(); self.critic_2_opt.zero_grad()
        critic_loss.backward()
        self.critic_1_opt.step(); self.critic_2_opt.step()

        # ---- delayed actor + target update ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)
            # BRAC policy regularization: pull the acting policy to the dataset action
            bc_mse = ((pi - actions) ** 2).sum(-1)
            # TD3+BC value normalization: balance RL vs imitation across reward scales
            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / (q.abs().mean().detach() + 1e-8)
            actor_loss = (self.beta_1 * bc_mse - lmbda * q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # soft target updates
            for net, tgt in ((self.critic_1, self.critic_1_target),
                             (self.critic_2, self.critic_2_target),
                             (self.actor, self.actor_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

So the causal chain, start to finish. I started stuck: off-policy actor-critic on a frozen dataset overestimates out-of-distribution actions, the maximizing policy chases the overestimate off-data, and the value diverges. TD3 gives me the right online overestimation controls — clipped-min twin critics, target smoothing, delayed updates — but none of them keep the policy on the data, so offline they're necessary but not sufficient. The minimalist fix is a behavior-cloning penalty on the actor with the value term normalized by `1/mean|Q|` so one coefficient transfers across reward scales — that's the floor, and it works, but it only guards the action I take, not the next-action the critic bootstraps from, so the overestimation leak reopens one step downstream in the target. Recognizing the two penalty locations of the behavior-regularized framework, I add the missing half — a value penalty inside the critic target — and because my policy is deterministic, the divergence collapses from KL/MMD-against-a-learned-behavior-model down to a plain squared distance to the dataset's own recorded next action, no extra network. The framework had coupled the two penalty strengths, but actor-side and critic-side caution are different jobs, so I decouple them into `beta_1` and `beta_2` and tune per environment. Then I refuse to inherit the architecture bundle blindly: LayerNorm goes in the critic because it provably bounds `|Q| <= ||w_head||` on out-of-distribution actions and thus caps the exact extrapolation that drives the loop, but not in the actor, whose output is already bounded and pulled to data; depth goes to three layers for capacity on a large static dataset; batch and learning rate go up on dense locomotion for faster low-variance convergence but stay small on sparse tasks where they hurt; the discount goes to `0.999` on long sparse-reward tasks because `0.99^1000` arithmetically erases the terminal signal while `0.999^1000` preserves a third of it; and I drop the inherited state normalization to keep the method online-compatible at negligible offline cost. The whole thing is TD3 plus two squared-distance penalties and a handful of read-off design choices, every one individually justified and removable.
