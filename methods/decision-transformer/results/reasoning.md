OK, let me think this through from scratch. I have a fixed pile of logged trajectories — states, actions, rewards — from some grab-bag of policies, mostly mediocre, a few lucky stretches of competence. No simulator to poke, no way to collect a single new transition. I want a controller out of this that behaves well. And the thing nagging at me before I write a line of code is that the two obvious recipes both fail in opposite directions, and I want to understand *why* precisely, because the failure mode is going to tell me what to build.

Recipe one: just do RL the normal way. Fit a value function by temporal-difference learning, $Q(s_t,a_t)\leftarrow r_t+\gamma\max_{a'}Q(s_{t+1},a')$, then act greedily. The target is built from my own current estimate at the next state — I'm bootstrapping. Online this is fine because if my estimate is wrong, I go act, collect the real outcome, and the data corrects me. But here the data is frozen. So watch what happens: the $\max_{a'}$ pulls the target toward whatever action looks best under $Q$, and the actions that look best are very often ones the dataset barely covers, because the network is free to extrapolate optimistically into the gaps. The target inherits that optimism, the next sweep pushes $Q$ further up there, and there's no fresh rollout to ever slap it back down. Value overestimation, compounding, with the policy chasing it into exactly the region of action space I have no data on. That's the documented offline failure. And it sits on top of the deeper structural problem: function approximation, bootstrapping, and off-policy data together — the deadly triad — are known to diverge even before you add the offline twist. So the entire field's response has been to bolt on conservatism: constrain the policy to the data's support, or penalize $Q$ on out-of-distribution actions so it learns a pessimistic lower bound. Which works, but stare at what it is — it's a patch applied *because* we're optimizing a policy against a learned function that lies in the gaps. The conservatism exists to defend against the lie. The lie exists because we bootstrap.

Recipe two: forget value functions, just clone. Supervised learning, regress the logged action on the state, $\max_\theta\sum\log\pi_\theta(a\mid s)$. Beautifully stable — no bootstrapping, no triad, no overestimation, it's just regression. But it imitates the *average* policy that produced the data, so it's capped at mediocre. Garbage in, garbage out.

So I'm caught. Stability lives on the supervised side; the ability to exceed the data lives on the RL side; and the bridge everyone uses — bootstrapping a value function — is the source of every instability I listed. Let me say the thing I keep circling: bootstrapping is the disease. The deadly triad, the overestimation, the slow one-step-at-a-time credit propagation, the need for a discount factor that then makes the agent myopic, the conservatism band-aids — all of it is downstream of the Bellman backup feeding on its own estimates. What if I just... don't? What if I drop TD entirely and don't fit a value function at all?

But then I'm back to behavior cloning, which can't beat the data. Unless. Unless cloning isn't actually the only supervised thing I can do. Let me look harder at what I threw away.

Here's what bugs me about plain cloning: it learns $a\sim\pi(a\mid s)$, collapsing every trajectory that ever passed through state $s$ — the brilliant ones and the awful ones — into one conditional distribution. Of course it's average; I *averaged*. But the dataset isn't actually a single policy. It's a whole spectrum of behaviors, each of which achieved some outcome. The outcome information is sitting right there in the rewards and I'm discarding it. A given state $s$ appears in a trajectory that limped to a return of 3 and in another that stormed to a return of 90, and the action taken was different in each, *and that difference is exactly the signal I want*. If I could tell the model "you're in the version of this that ends up at 90," it should reproduce the 90-action.

So condition on the outcome. Train $a\sim\pi_\theta(a\mid s,\,\text{outcome})$, where for each logged action I label it with the return its trajectory actually went on to achieve. This is honest supervised learning — no bootstrapping, every label is a real number computed by summing rewards, nothing optimized against a learned function. And it's not capped at average, because now each action is filed under the return it produced, so the model learns the whole map from desired-return to behavior, not the average behavior. Then at deployment I just *ask* for a high return and read off the action. The dataset never had to contain an expert policy; it only had to contain, scattered across many trajectories, the good actions paired with the good outcomes they led to. Hindsight does the relabeling: every trajectory is a "successful" demonstration of achieving its *own* return.

What's the right "outcome" variable to feed, concretely? The total trajectory return is one number for the whole episode, but I'm making a decision at every timestep, and the relevant quantity at timestep $t$ isn't the return from the start — that's partly spent — it's the return I have *left to collect from here on*. So feed the return-to-go: $\widehat{R}_t=\sum_{t'=t}^{T} r_{t'}$, the sum of future rewards from $t$ onward. At $t=1$ it's the full episode return; as the episode proceeds it shrinks. And note carefully — I must feed the *future* return, not the past reward $r_t$. If I fed past rewards the model would condition on what already happened, which is useless for steering toward a target; what I need is for the action at $t$ to be conditioned on the future I'm aiming at. Returns-to-go, not rewards. That distinction is the whole game.

Now I've committed to: no value function, no TD, a supervised model that conditions each action on the state and the return I still want. The single-step version of this — condition on the current state and a desired return, predict the action — is basically the upside-down / reward-conditioned policy idea, and it works to a point. But $K=1$, conditioning on just the present state, throws away the trajectory's history, and something tells me history matters. Which makes me reframe the whole thing one more notch, and this is where it clicks: I don't have isolated $(s,\widehat{R})\to a$ examples, I have *trajectories* — ordered streams. I have GPT, which eats an ordered stream of tokens and learns the next one by plain maximum likelihood, no bootstrapping anywhere, stable at scale, and — the property I underlined earlier — its self-attention forms an association between any two positions in a single hop. So why am I treating this as tabular supervised learning at all? Treat the trajectory as a *sequence* and let a causal Transformer model it autoregressively. That's the move: RL as conditional sequence modeling.

Lay out the tokens. The natural per-timestep triple is the return-to-go, the state, and the action, and the order has to respect what predicts what. I want the action at $t$ to be generated after the model has seen the return I'm targeting and the state I'm in, so the layout is

$$\tau=\big(\widehat{R}_1,\,s_1,\,a_1,\;\widehat{R}_2,\,s_2,\,a_2,\;\dots,\;\widehat{R}_T,\,s_T,\,a_T\big).$$

Three tokens per timestep. Run them through GPT with a causal mask so token $i$ attends only to tokens $\le i$. Then the hidden state sitting on the *state* token $s_t$ has seen everything up to and including $\widehat{R}_t$ and $s_t$ — the whole history plus the current desired return and current state — and I attach the action head there to predict $a_t$. Concretely the model learns $p_\theta\!\left(a_t\mid \widehat{R}_1,s_1,a_1,\dots,\widehat{R}_t,s_t\right)$. That's exactly $a\sim\pi(a\mid \text{history},\,s_t,\,\widehat{R}_t)$ — the return-conditioned policy I derived, but now with the full preceding context for free, because that's just what causal attention gives you.

The training objective is then the most boring thing imaginable, which is the point — it's why all the instability is gone. Sample sub-sequences of $K$ timesteps from the offline trajectories ($3K$ tokens). Run the forward pass. Take the prediction at each state token and push it toward the logged action: cross-entropy if actions are discrete, mean-squared error if they're continuous, averaged over the timesteps in the window. No target network, no Bellman residual, no max over actions, no value function to overestimate. Just "given the return-to-go and the history, what action was taken." I tried for a second to also predict the next state and the next return-to-go as auxiliary targets — it's permissible and trivial to add extra heads — but on the locomotion and Atari data it didn't help the action quality, so I keep the loss on actions only and leave the state/return heads as optional. (One place it *is* useful: if I want the model to also be a critic, I let it predict $\widehat{R}$ and even learn the initial return distribution $p(\widehat{R}_1)$, and I'll come back to that.)

Now the deployment procedure, which is where the conditioning actually does its work. I pick a target return-to-go $\widehat{R}_1$ and set it high — this is the prompt, the "which skill do you want" knob. I observe the start state $s_1$. I feed $(\widehat{R}_1,s_1)$, the model emits $a_1$, I execute it. The environment hands me reward $r_1$ and the next state $s_2$. Here's the crucial step: I have to keep the return-to-go meaning what it means during training — "the return still to be collected from here." So I decrement: $\widehat{R}_2=\widehat{R}_1-r_1$. Let me sanity-check that against the definition. By construction $\widehat{R}_t=\sum_{t'=t}^{T}r_{t'}=r_t+\sum_{t'=t+1}^{T}r_{t'}=r_t+\widehat{R}_{t+1}$, so $\widehat{R}_{t+1}=\widehat{R}_t-r_t$. Yes — peel off the reward I just received and what remains is the return I still want. That recursion is the engine of the rollout: I'm not predicting the return, I'm *accounting* for it, subtracting each received reward from the budget so the model is always conditioned on a target that's consistent with how it saw returns-to-go at training time. Then I feed the whole context $(\widehat{R}_1,s_1,a_1,\widehat{R}_2,s_2)$, get $a_2$, execute, decrement again, and so on, cropping the context to the last $K$ timesteps as it grows. The model and the environment jointly generate the trajectory: the model proposes an action consistent with the remaining target, the environment realizes a reward, and the target updates.

Let me make sure I believe the central claim — that this lets a model trained on mixed junk act well — because if I don't believe it, none of this is worth coding. Three things have to be true. First, ceiling: can it exceed the average trajectory? Yes, because conditioning breaks the averaging — prompting a high $\widehat{R}_1$ selects the slice of the behavior map associated with high returns, the good actions that *were* in the data but scattered across the better moments of many trajectories. Second, and stronger — can it produce behavior *no single trajectory* contains? Take the shortest-path-on-a-graph toy: reward $-1$ per step, $0$ at the goal, so return-to-go is just negative remaining path length and "maximize it" means "take the fewest steps." Train only on random walks — no demonstration ever solves it well. But each action is conditioned on the *local* state and the *local* return-to-go, and the model has seen, across thousands of random walks, lots of little segments: "from node $u$, with this much budget, going to $v$ tends to be followed by this." At test time I prompt the largest realizable return and let it generate; it can *stitch* — chain a good segment from one walk onto a good segment from another — because nothing in the objective tied a segment to its original trajectory; the segments are interchangeable conditioned on state and return-to-go. So it can sequence an optimal path that appears in no training trajectory. That's policy improvement, and it came out of plain supervised sequence modeling plus hindsight return labels — no dynamic programming, no Bellman backup anywhere.

Third — credit assignment under sparse or delayed reward, which is where I most expect TD to die and where I want the Transformer to shine. Think about Key-to-Door: grab a key in room one, wander an irrelevant room two, and only at the door in room three do you get a reward, and only if you took the key. The signal has to connect an action at the very start to an outcome at the very end, skipping everything in the middle. TD propagates that reward backward one Bellman step per sweep, so it has to crawl across the whole empty middle, and any distractor reward in between corrupts the crawl — this is exactly the regime where value learning collapses. But self-attention doesn't crawl. The query at the door token can directly attend to the key token in one hop; the dot-product between query and key vectors forms the state-return association directly, no matter the distance. And because I fed return-to-go tokens, the model has a clean handle on "this episode is heading for reward 1" versus "0," so it learns which early events the eventual return hinges on. I'd expect, if I peeked at the attention, to see the model attending right at the pivotal events — picking up the key, reaching the door — and if I let it also predict the running return, to see that predicted return jump precisely when the key is grabbed. That's the credit assignment, done by association rather than backup.

And notice what fell away on its own. No discount factor — the return-to-go is an undiscounted sum, $\gamma=1$, and I don't need $\gamma<1$ to make anything a contraction because there's no fixed-point iteration here, it's one supervised pass; which is good, because $\gamma<1$ was only ever introduced for convergence and finiteness and it bought me myopia I didn't want. No conservatism, no pessimism term, no behavior regularizer — and I can finally say *why* cleanly: those exist to stop a policy from exploiting the errors of a learned value function it's being optimized against. I never optimize against a learned function. My objective is regression to actions that genuinely occurred in the data; there's no learned critic to game, so there's nothing to be conservative about. The whole apparatus of offline-RL safety machinery is unnecessary because the thing it was guarding against — bootstrapped value optimization — isn't in the design.

One thing I should pressure-test: is this just behavior cloning on the high-return subset in disguise? Imagine the honest competitor — percentile behavior cloning, clone only the top-$X\%$ of trajectories by return. When data is plentiful that's actually a strong baseline and the model is, in part, learning to behave like the good slice. But two differences matter. Choosing the right percentile requires evaluating in the environment, which offline I cannot do; the return-conditioning sidesteps that by letting one model span all percentiles and selecting at test time via the prompt. And when data is scarce, throwing away all but the top $X\%$ is wasteful — the model trained on the *whole* distribution can use the structure in the mediocre trajectories (and the stitching) to generalize better than one trained on a thin good slice. So it's related to clone-the-best but strictly more flexible, and it's the stitching that makes it more than imitation.

Let me also settle the architectural choices before coding, because a couple of them are non-obvious. The three modalities — return-to-go (a scalar), state (a vector, or an image), action (discrete index or continuous vector) — live in completely different spaces and scales. A single shared input projection would smear them together, so I give each its own embedding: a linear map per modality to the hidden width, and for image states a convolutional encoder instead of the linear map. Then a layer norm on the embeddings to keep the scales sane going into attention.

Positional information is the subtle one. The standard GPT positional embedding tags a token by its index in the sequence, $0,1,2,\dots$ But my sequence runs three tokens per environment timestep, so positions $1,2,3$ are all timestep $1$ and I don't actually care that the action is "two after" the return; I care that all three belong to env-timestep $1$. So I drop the per-token positional embedding and instead learn a per-*timestep* embedding and add the *same* timestep vector to all three tokens of that timestep. That tells the model "these three are the same moment" while letting attention sort out the within-moment order via the causal mask. Add, don't concatenate, to keep the width fixed.

Why the action head on the state token and not elsewhere — because that's the position whose causal context is exactly $(\text{history},\widehat{R}_t,s_t)$, the conditioning I derived. The head is a linear map to action logits for discrete actions (softmax + cross-entropy), or a linear map for continuous actions; for bounded continuous actions I squash with a tanh so the output lives in the valid range.

Context length $K$: I claimed history helps, so $K=1$ should be visibly worse — and it is the natural ablation to run, dropping the model to a single timestep. My intuition for *why* longer context helps even when frame-stacking already gives Markovian-ish state: I'm modeling a whole *distribution* of policies, not one, and the recent context lets the model infer which policy regime generated the surrounding behavior, which sharpens the action prediction. So I keep $K$ at a few dozen timesteps. And I find I want a *larger* model than typical RL uses — RL fits a single policy, but I'm fitting the distribution over returns and policies, which is a richer object, so the extra capacity earns its keep. The rest is inherited language-model hygiene: AdamW, weight decay on the linear/conv weights but not on biases / layer-norms / embeddings, a learning-rate warmup then decay, gradient clipping — none of it method-specific, all of it free from the Transformer literature, which is precisely the dividend of casting RL as sequence modeling: I get to ride a mature, stable training stack instead of fighting RL-specific divergence.

Now let me write it, grounded in a clean GPT stack. First the model — embed each modality, add the per-timestep encoding, interleave into the $(\widehat{R}_1,s_1,a_1,\dots)$ order, run the causal Transformer, read the action prediction off the state-token positions.

```python
import torch
import torch.nn as nn
import transformers  # GPT-2 stack; we only use it as a causal transformer over embeddings

class DecisionTransformer(nn.Module):
    """Models (R_1, s_1, a_1, R_2, s_2, a_2, ...) with a causal GPT and predicts actions."""
    def __init__(self, state_dim, act_dim, hidden_size,
                 max_length=None, max_ep_len=4096, action_tanh=True, **kwargs):
        super().__init__()
        self.state_dim, self.act_dim = state_dim, act_dim
        self.hidden_size, self.max_length = hidden_size, max_length

        # a GPT-2 with the built-in positional embedding disabled -- we supply our own
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = transformers.GPT2Model(config)

        # one embedding per modality: scalar return, state vector, action vector
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state  = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        # per-timestep (NOT per-token) positional encoding: one moment == three tokens
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # action head; tanh to keep bounded continuous actions in range
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim), *([nn.Tanh()] if action_tanh else []))

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        B, K = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((B, K), dtype=torch.long)

        # embed each modality, then add the SAME timestep vector to all three of a moment
        t = self.embed_timestep(timesteps)
        s = self.embed_state(states)        + t
        a = self.embed_action(actions)      + t
        R = self.embed_return(returns_to_go) + t

        # interleave to (R_1, s_1, a_1, R_2, s_2, a_2, ...): 3K tokens
        tokens = torch.stack((R, s, a), dim=1).permute(0, 2, 1, 3).reshape(B, 3 * K, self.hidden_size)
        tokens = self.embed_ln(tokens)
        mask = torch.stack((attention_mask,) * 3, dim=1).permute(0, 2, 1).reshape(B, 3 * K)

        # feed embeddings (not token ids) through the causal transformer
        h = self.transformer(inputs_embeds=tokens, attention_mask=mask)['last_hidden_state']
        # un-interleave: dim 1 indexes modality -- 0:R, 1:s, 2:a
        h = h.reshape(B, K, 3, self.hidden_size).permute(0, 2, 1, 3)

        # predict a_t from the STATE token: its causal context is (history, R_t, s_t)
        return self.predict_action(h[:, 1])

    def get_action(self, states, actions, returns_to_go, timesteps):
        # deployment: keep only the last K timesteps, pad on the left, return the last action
        states = states.reshape(1, -1, self.state_dim)[:, -self.max_length:]
        actions = actions.reshape(1, -1, self.act_dim)[:, -self.max_length:]
        returns_to_go = returns_to_go.reshape(1, -1, 1)[:, -self.max_length:]
        timesteps = timesteps.reshape(1, -1)[:, -self.max_length:]
        # (left-padding + attention mask omitted here for clarity; see rollout)
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0, -1]
```

The training step is just masked regression onto the logged actions — note `rtg[:, :-1]` so the return-to-go token aligns as the *input* before each state, never leaking the action's own future:

```python
def train_step(model, batch, optimizer):
    states, actions, rewards, dones, rtg, timesteps, mask = batch
    target = actions.clone()
    action_preds = model.forward(states, actions, rtg[:, :-1], timesteps, attention_mask=mask)

    # keep only real (non-padded) positions, then MSE for continuous actions
    act_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
    target       = target.reshape(-1, act_dim)[mask.reshape(-1) > 0]
    loss = torch.mean((action_preds - target) ** 2)   # cross-entropy instead, for discrete actions

    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    return loss.item()
```

And the rollout — this is the return-to-go recursion made literal. Prompt a high target, act, subtract the received reward from the target, repeat:

```python
def rollout(env, model, target_return, max_ep_len, scale, state_dim, act_dim, device):
    state = env.reset()
    states = torch.from_numpy(state).reshape(1, state_dim).float().to(device)
    actions = torch.zeros((0, act_dim), device=device)
    target = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)  # scaled R_1
    timesteps = torch.tensor(0, device=device).reshape(1, 1)

    ep_return = 0
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)  # placeholder for a_t
        action = model.get_action(states, actions, target, timesteps)  # conditioned on current R-to-go
        actions[-1] = action
        state, reward, done, _ = env.step(action.detach().cpu().numpy())

        states = torch.cat([states, torch.from_numpy(state).reshape(1, state_dim).float().to(device)], dim=0)
        # THE recursion: remaining target = previous target - reward just received (scaled)
        new_target = target[0, -1] - reward / scale
        target = torch.cat([target, new_target.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        ep_return += reward
        if done:
            break
    return ep_return
```

For discrete actions and image states the only changes are: a convolutional encoder in place of `embed_state`, the action head emits logits with a cross-entropy loss, and the action embedding is a lookup rather than a linear map; the token layout, the per-timestep encoding, the state-token action head, and the decrement-the-return rollout are identical.

So the whole chain, in one breath: bootstrapping is the source of every instability in offline value learning — the deadly triad, the overestimation, the slow distractable credit assignment, the discount-induced myopia, the conservatism patches that all exist to defend against optimizing a policy against a learned value function. Drop bootstrapping entirely and the stable alternative is supervised learning, but plain cloning averages away the good behavior. Recover the good behavior by conditioning each action on the *outcome* — the return-to-go, the future reward still to collect — and relabeling every logged action with the return it actually achieved, which is honest supervision with no learned critic. Treat the trajectory as a token stream $(\widehat{R}_1,s_1,a_1,\dots)$ and a causal Transformer turns this into next-action prediction with full history for free, and its single-hop attention does long-range credit assignment that Bellman backups can't. At test time, prompt the return you want, act, subtract each reward from the target via $\widehat{R}_{t+1}=\widehat{R}_t-r_t$, and the model and environment co-generate a trajectory that hits the conditioned return — stitching optimal behavior out of mediocre data, with nothing but cross-entropy/MSE and a GPT.
