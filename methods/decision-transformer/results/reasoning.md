OK, let me think this through from scratch. I have a fixed pile of logged trajectories — states, actions, rewards — from some grab-bag of policies, mostly mediocre, a few lucky stretches of competence. No simulator to poke, no way to collect a single new transition. I want a controller out of this that behaves well. And the thing nagging at me before I write a line of code is that the two obvious recipes both fail in opposite directions, and I want to understand *why* precisely, because the failure mode is going to tell me what to build.

Recipe one: just do RL the normal way. Fit a value function by temporal-difference learning, $Q(s_t,a_t)\leftarrow r_t+\gamma\max_{a'}Q(s_{t+1},a')$, then act greedily. The target is built from my own current estimate at the next state — I'm bootstrapping. Online this is fine because if my estimate is wrong, I go act, collect the real outcome, and the data corrects me. But here the data is frozen. So watch what happens: the $\max_{a'}$ pulls the target toward whatever action looks best under $Q$, and the actions that look best are very often ones the dataset barely covers, because the network is free to extrapolate optimistically into the gaps. The target inherits that optimism, the next sweep pushes $Q$ further up there, and there's no fresh rollout to ever slap it back down. Value overestimation, compounding, with the policy chasing it into exactly the region of action space I have no data on. That's the documented offline failure. And it sits on top of the deeper structural problem: function approximation, bootstrapping, and off-policy data together — the deadly triad — are known to diverge even before you add the offline twist. So the entire field's response has been to bolt on conservatism: constrain the policy to the data's support, or penalize $Q$ on out-of-distribution actions so it learns a pessimistic lower bound. Which works, but stare at what it is — it's a patch applied *because* we're optimizing a policy against a learned function that lies in the gaps. The conservatism exists to defend against the lie. The lie exists because we bootstrap.

Recipe two: forget value functions, just clone. Supervised learning, regress the logged action on the state, $\max_\theta\sum\log\pi_\theta(a\mid s)$. Beautifully stable — no bootstrapping, no triad, no overestimation, it's just regression. But it imitates the *average* policy that produced the data, so it's capped at mediocre. Garbage in, garbage out.

So I'm caught. Stability lives on the supervised side; the ability to exceed the data lives on the RL side; and the bridge everyone uses — bootstrapping a value function — is the source of every instability I listed. Let me trace where each problem comes from, because if they all trace back to one place I want to know. The overestimation: comes from the $\max_{a'}$ over a bootstrapped target. The deadly triad: bootstrapping is one of its three named ingredients. The slow one-step-at-a-time credit propagation: that's the Bellman backup moving reward one step per sweep. The discount factor's myopia: $\gamma<1$ is there to make the Bellman operator a contraction so the fixed-point iteration converges. The conservatism patches: they exist to stop a policy from exploiting errors in a learned value function. Every single one of these has a value function being bootstrapped in its causal chain. So the natural question is whether I can drop TD entirely and not fit a value function at all — and whether anything I actually need survives that amputation.

But then I'm back to behavior cloning, which can't beat the data. Unless. Unless cloning isn't actually the only supervised thing I can do. Let me look harder at what I threw away.

Here's what bugs me about plain cloning: it learns $a\sim\pi(a\mid s)$, collapsing every trajectory that ever passed through state $s$ — the brilliant ones and the awful ones — into one conditional distribution. Of course it's average; I *averaged*. But the dataset isn't actually a single policy. It's a whole spectrum of behaviors, each of which achieved some outcome. The outcome information is sitting right there in the rewards and I'm discarding it. A given state $s$ appears in a trajectory that limped to a return of 3 and in another that stormed to a return of 90, and the action taken was different in each, *and that difference is exactly the signal I want*. If I could tell the model "you're in the version of this that ends up at 90," it should reproduce the 90-action.

So condition on the outcome. Train $a\sim\pi_\theta(a\mid s,\,\text{outcome})$, where for each logged action I label it with the return its trajectory actually went on to achieve. This is honest supervised learning — no bootstrapping, every label is a real number computed by summing rewards, nothing optimized against a learned function. The claim I have to be careful about is "not capped at average." Plain cloning at state $s$ learns $\mathbb{E}[a\mid s]$, averaging over all trajectories through $s$. With conditioning I instead learn $\mathbb{E}[a\mid s,\,\text{outcome}=R]$, a separate prediction per value of $R$. So the question is whether $\mathbb{E}[a\mid s,\,R{=}\text{high}]$ actually picks out the good action rather than the average one — and that depends entirely on whether, in the data, the high-$R$ trajectories through $s$ took systematically different actions than the low-$R$ ones. If they did, conditioning separates them; if every return through $s$ used the same action, conditioning buys nothing and I'm back to the average. I'll need to convince myself the first case is the typical one, ideally on a concrete instance, before I trust this. Granting it for now: each action is filed under the return it produced, so the model learns a map from desired-return to behavior, and at deployment I *ask* for a high return and read off the action. The dataset never had to contain an expert policy; it only had to contain, scattered across many trajectories, the good actions paired with the good outcomes they led to. Hindsight does the relabeling: every trajectory is a "successful" demonstration of achieving its *own* return.

What's the right "outcome" variable to feed, concretely? The total trajectory return is one number for the whole episode, but I'm making a decision at every timestep, and the relevant quantity at timestep $t$ isn't the return from the start — that's partly spent — it's the return I have *left to collect from here on*. So feed the return-to-go: $\widehat{R}_t=\sum_{t'=t}^{T} r_{t'}$, the sum of future rewards from $t$ onward. At $t=1$ it's the full episode return; as the episode proceeds it shrinks. And I have to feed the *future* return, not the past reward $r_t$: if I fed past rewards the model would condition on what already happened, which says nothing about which action steers toward a target. The action at $t$ should be conditioned on the future I'm aiming at, so it has to be returns-to-go, not rewards.

Now I've committed to: no value function, no TD, a supervised model that conditions each action on the state and the return I still want. The single-step version of this — condition on the current state and a desired return, predict the action — is the upside-down / reward-conditioned policy idea, and it works to a point. But that version conditions on just the present state and throws away the trajectory's history. Does history matter? Two reasons to suspect it does. First, the data is a *mixture* of policies; if a recent stretch of context can reveal which policy regime I'm inside, the action prediction sharpens, the way a few words of context disambiguate a sentence. Second, in partially-observed or sparse-reward tasks the relevant signal may sit several steps back, and a single-state model has no slot to carry it. So I'd rather not throw the history away.

But that reframes what kind of object I'm actually modeling. I don't have isolated $(s,\widehat{R})\to a$ examples; I have *trajectories* — ordered streams of $(\widehat{R},s,a)$ triples. There's a class of model built precisely for ordered streams: GPT eats a token sequence and learns the next token by plain maximum likelihood — no bootstrapping anywhere, stable at scale — and its self-attention can form an association between any two positions in a single hop. If I lay the trajectory out as a token stream and let a causal Transformer model it autoregressively, then the next-action prediction automatically conditions on the *entire* preceding stream, not just the current state. The history comes for free, as a property of causal attention, rather than as something I have to engineer. That is enough of a reason to stop treating this as tabular supervised learning and treat the trajectory as a sequence.

Lay out the tokens. The natural per-timestep triple is the return-to-go, the state, and the action, and the order has to respect what predicts what. I want the action at $t$ to be generated after the model has seen the return I'm targeting and the state I'm in, so the layout is

$$\tau=\big(\widehat{R}_1,\,s_1,\,a_1,\;\widehat{R}_2,\,s_2,\,a_2,\;\dots,\;\widehat{R}_T,\,s_T,\,a_T\big).$$

Three tokens per timestep. Run them through GPT with a causal mask so token $i$ attends only to tokens $\le i$. Then the hidden state sitting on the *state* token $s_t$ has seen everything up to and including $\widehat{R}_t$ and $s_t$ — the whole history plus the current desired return and current state — and if I attach the action head there it predicts $a_t$ from exactly that context. The model learns $p_\theta\!\left(a_t\mid \widehat{R}_1,s_1,a_1,\dots,\widehat{R}_t,s_t\right)$, which is the return-conditioned policy $a\sim\pi(a\mid \text{history},\,s_t,\,\widehat{R}_t)$ I was reaching for, now carrying the full preceding context. The conditioning I wanted and the history I wanted both land on the same hidden state.

The training objective is then the most boring thing imaginable, which is the point — it's why all the instability is gone. Sample sub-sequences of $K$ timesteps from the offline trajectories ($3K$ tokens). Run the forward pass. Take the prediction at each state token and push it toward the logged action: cross-entropy if actions are discrete, mean-squared error if they're continuous, averaged over the real, non-padded timesteps in the window. No target network, no Bellman residual, no max over actions, no value function to overestimate. Just "given the return-to-go and the history, what action was taken." I can leave next-state and next-return heads in the module because once the action token is visible those predictions are causally well-formed, but the control objective I need is the action loss; the auxiliary heads do not have to carry the policy.

Now the deployment procedure. I pick a target return-to-go $\widehat{R}_1$ and set it high — this is the prompt, the "which skill do you want" knob. I observe the start state $s_1$. I feed $(\widehat{R}_1,s_1)$, the model emits $a_1$, I execute it. The environment hands me reward $r_1$ and the next state $s_2$. Now I need a $\widehat{R}_2$ to feed, and it has to mean what it meant during training: "the return still to be collected from here." The obvious move is to decrement, $\widehat{R}_2=\widehat{R}_1-r_1$, but I should check that this actually reproduces the training-time labels rather than just feeling right. From the definition, $\widehat{R}_t=\sum_{t'=t}^{T}r_{t'}=r_t+\sum_{t'=t+1}^{T}r_{t'}=r_t+\widehat{R}_{t+1}$, so algebraically $\widehat{R}_{t+1}=\widehat{R}_t-r_t$. Let me also run a concrete sequence through both definitions, because off-by-one errors hide in exactly this kind of recursion. Take rewards $(2,-1,3,0)$. By the summation definition the returns-to-go are $\big(\sum_{0}r,\sum_{1}r,\sum_{2}r,\sum_{3}r\big)=(4,2,3,0)$. By the rollout recursion, starting from $\widehat{R}_1=4$ and subtracting each reward: $4\to4-2=2\to2-(-1)=3\to3-3=0$, giving $(4,2,3,0)$. They agree term for term — so peeling off each received reward from the budget really does regenerate the same labels the model was trained on, including across the negative reward where I'd most fear a sign slip. I'm not predicting the return, I'm *accounting* for it. Then I feed the whole context $(\widehat{R}_1,s_1,a_1,\widehat{R}_2,s_2)$, get $a_2$, execute, decrement again, and so on, cropping the context to the last $K$ timesteps as it grows. The model and the environment jointly generate the trajectory: the model proposes an action consistent with the remaining target, the environment realizes a reward, and the target updates.

I still owe myself the argument that a model trained on mixed junk can act *above* the data's average, and earlier I flagged that this rests on whether high-return and low-return trajectories through a shared state actually took different actions. Let me stop asserting it and build the smallest example that settles it. Shortest path on a graph: reward $-1$ per step, $0$ at the goal $G$, so the return-to-go at a node is the negative number of remaining steps, and "want a high return-to-go" means "want the fewest steps left." Train only on random walks. Take two of them:

- $w_1:\ A\to C\to A\to B\to G$
- $w_2:\ D\to B\to A\to C\to G$

Relabel every step with the state, the return-to-go (negative steps remaining *in that walk*), and the action taken (the next node). Doing this by hand for the steps at the shared node $C$: in $w_1$, from $C$ it takes three more transitions ($C\to A\to B\to G$) to reach the goal, so its token is $(\text{state}{=}C,\ \widehat{R}{=}-3,\ a{=}A)$; in $w_2$, from $C$ it is one transition ($C\to G$) to the goal, so its token is $(\text{state}{=}C,\ \widehat{R}{=}-1,\ a{=}G)$. Now query the model the way deployment will: I'm at $C$ and I prompt the *good* return-to-go, $\widehat{R}=-1$ (one step left). The only training token matching $(C,\widehat{R}{=}-1)$ says $a=G$ — the direct $C\to G$ step. If instead I prompted the bad $\widehat{R}=-3$, the matching token says $a=A$, the wandering step. So at the *same* state, conditioning on the return-to-go cleanly separates the good action from the bad one — and crucially the good action $C\to G$ was demonstrated only inside $w_2$, a meandering walk that started all the way back at $D$, while the query that surfaces it could just as well have arrived at $C$ via $w_1$'s prefix. Nothing in the supervised objective tied the $C\to G$ step to the rest of $w_2$; once I condition on state and return-to-go, the segment is reusable. That is stitching, and it is the mechanism that lets recombination beat the average trajectory — verified on a case I can check by hand, not assumed.

But the same hand-trace shows the limit, which I want to be honest about. Suppose neither walk ever takes the direct $C\to G$ step — say I replace $w_2$ with $D\to B\to A\to B\to G$. Then there is *no* training token with $(C,\widehat{R}{=}-1)$ at all; the only token at $C$ is $(C,\widehat{R}{=}-3,a{=}A)$. Prompting $\widehat{R}=-1$ at $C$ now conditions on something the data never demonstrated, and the model has nothing to stitch — conditioning cannot conjure an action whose segment never appears anywhere in the data. So the improvement is real but bounded: it recombines good fragments that *exist*, scattered across trajectories; it does not invent them. That's a fair characterization of the mechanism, and it's exactly the regime offline RL lives in.

A practical wrinkle on this toy: if I also want the model to *generate* the return token (rather than me supplying it), I can bias generation toward shorter paths with a simple prior over the encoded remaining-length token, $P_{\mathrm{prior}}(\widehat{R}=k)\propto T+1-k$, multiplying the model probability by $P_{\mathrm{prior}}(\widehat{R}_t)^{10}$ before sampling the return token. The action model still does the stitching; the prior just steers which return-to-go I commit to. The whole improvement here is conditional generation plus hindsight return labels — no dynamic programming, no Bellman backup.

The sparse-reward case is where I most expect TD to struggle. Think about Key-to-Door: grab a key in room one, wander an irrelevant room two, and only at the door in room three do you get a reward, and only if you took the key. The signal has to connect an action at the very start to an outcome at the very end, skipping everything in the middle. TD propagates that reward backward one Bellman step per sweep, so it has to crawl across the whole empty middle — order $N$ sweeps for an $N$-step gap — and any distractor reward in between can corrupt the crawl. Self-attention has a different cost structure: the query at the door token can attend to the key token in one hop, the dot-product between query and key vectors forming the long-range association in a single layer regardless of distance. So on paper the sparse, long-horizon credit problem that costs TD $O(N)$ propagation costs attention $O(1)$ hops. That's the reason to *expect* an advantage; I shouldn't call it confirmed from the architecture alone, because whether the model learns to *use* that one-hop path is an empirical question, not a guarantee of attention's existence.

But I can name a concrete, checkable prediction that would tell me whether the mechanism is really operating. Because I feed return-to-go tokens, the model carries an explicit handle on "this episode is heading for terminal reward 1" versus "0." If I keep the auxiliary return-prediction head and read out its estimate of the eventual return as the prefix grows, then in a successful model that estimate should jump at the timestep where the key is taken and stay flat across the empty middle room — the return head should localize the pivotal event. That is a thing I can actually plot once trained, and if the jump does not align with the key-grab, my credit-assignment story is wrong. So I'll treat "attention does the long-range credit assignment by association rather than backup" as a hypothesis with a built-in test, not as something the design already proves.

And notice what fell away on its own. No discount factor — the return-to-go is an undiscounted sum, $\gamma=1$, and I don't need $\gamma<1$ to make anything a contraction because there's no fixed-point iteration here, it's one supervised pass; which is good, because $\gamma<1$ was introduced for convergence and finiteness and it can buy me myopia I don't want. No conservatism, no pessimism term, no behavior regularizer in the objective. Those terms exist to stop a policy from exploiting the errors of a learned value function it's being optimized against. I never optimize a policy against a learned critic. My objective is regression to actions that genuinely occurred in the data, conditioned by the desired outcome, so the specific overestimation loop that conservatism defends against is not in the design.

One thing I should pressure-test: is this just behavior cloning on the high-return subset in disguise? Imagine the honest competitor — percentile behavior cloning, clone only the top-$X\%$ of trajectories by return. When data is plentiful that's actually a strong baseline and the model is, in part, learning to behave like the good slice. But two differences matter. Choosing the right percentile requires evaluating in the environment, which offline I cannot do; the return-conditioning sidesteps that by letting one model span all percentiles and selecting at test time via the prompt. And when data is scarce, throwing away all but the top $X\%$ is wasteful — the model trained on the *whole* distribution can use the structure in the mediocre trajectories (and the stitching) to generalize better than one trained on a thin good slice. So it's related to clone-the-best but strictly more flexible, and it's the stitching that makes it more than imitation.

Let me also settle the architectural choices before coding, because a couple of them are non-obvious. The three modalities — return-to-go (a scalar), state (a vector, or an image), action (discrete index or continuous vector) — live in completely different spaces and scales. A single shared input projection would smear them together, so I give each its own embedding: a linear map per modality to the hidden width, and for image states a convolutional encoder instead of the linear map. Then a layer norm on the embeddings to keep the scales sane going into attention.

Positional information is the subtle one. The standard GPT positional embedding tags a token by its index in the sequence, $0,1,2,\dots$ But my sequence runs three tokens per environment timestep, so positions $1,2,3$ are all timestep $1$. I do not only need the model to know that the action token is two slots after the return token; I need it to know that all three tokens belong to env-timestep $1$. In the vector-state version I can remove the built-in token-position table and learn a per-*timestep* embedding, adding the *same* timestep vector to all three tokens of that timestep. That tells the model "these three are the same moment" while letting attention sort out the within-moment order via the causal mask. Add, don't concatenate, to keep the width fixed.

Why the action head on the state token and not elsewhere — because that's the position whose causal context is exactly $(\text{history},\widehat{R}_t,s_t)$, the conditioning I derived. The head is a linear map to action logits for discrete actions (softmax + cross-entropy), or a linear map for continuous actions; for bounded continuous actions I squash with a tanh so the output lives in the valid range.

This is also the place where the implementation is most likely to be silently wrong, so I want to trace the index bookkeeping explicitly rather than trust it. I'll build the stacked sequence by stacking the three per-timestep embeddings, permuting, and reshaping to length $3T$, and the claim is that after running the transformer and reshaping *back* to $(T,3,\cdot)$, index $0$ holds return positions, index $1$ holds state positions, index $2$ holds action positions — so the action head must read index $1$. Let me check that with a tiny tagged example: tag every return embedding with the value $1$, every state with $2$, every action with $3$, and push $T=2$ timesteps through the exact stack/permute/reshape. The flattened sequence comes out as $[1,2,3,1,2,3]$ — i.e. $(\widehat{R}_1,s_1,a_1,\widehat{R}_2,s_2,a_2)$, the interleaving I wanted. Reshaping back to $(T,3)$ and slicing: index $0$ recovers $[1,1]$ (returns), index $1$ recovers $[2,2]$ (states), index $2$ recovers $[3,3]$ (actions). So the action prediction does have to be read off slice index $1$ — which matches `action_preds = self.predict_action(x[:, 1])` in the code, and the next-state and next-return heads correctly sit on slice $2$, the action-token positions, where the action has already been seen. Worth the thirty-second check, because a transposed index here would train the action head on the wrong conditioning context and the whole conditioning argument would be quietly broken while the loss still looked fine.

Context length $K$: if history is really doing work, the clean stress test is $K=1$, dropping the model to a single timestep. My intuition for why longer context helps even when frame-stacking already gives Markovian-ish state is that I'm modeling a whole distribution of policies, not one, and the recent context lets the model infer which policy regime generated the surrounding behavior, which sharpens the action prediction. So I keep $K$ at a few dozen timesteps, shorter for short goal-conditioned episodes. I also budget more capacity than a typical single-policy RL network, because the model has to represent a distribution over returns and behaviors rather than one narrow controller. The rest is inherited language-model hygiene: AdamW, weight decay on the linear/conv weights but not on biases / layer-norms / embeddings, a learning-rate warmup then decay, gradient clipping — none of it method-specific, all of it free from the Transformer literature, which is precisely the dividend of casting RL as sequence modeling: I get to ride a mature, stable training stack instead of fighting RL-specific divergence.

The code has to preserve exactly that causal order: embed each modality, add the per-timestep encoding, interleave into the $(\widehat{R}_1,s_1,a_1,\dots)$ order, run the causal Transformer, read the action prediction off the state-token positions, train only the masked action loss, and update the target return during rollout.

```python
import numpy as np
import torch
import torch.nn as nn
import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            pad = self.max_length - states.shape[1]
            attention_mask = torch.cat(
                [torch.zeros(pad, device=states.device), torch.ones(states.shape[1], device=states.device)]
            ).to(dtype=torch.long).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((1, pad, self.state_dim), device=states.device), states], dim=1
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((1, pad, self.act_dim), device=actions.device), actions], dim=1
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((1, pad, 1), device=returns_to_go.device), returns_to_go], dim=1
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((1, pad), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, **kwargs
        )
        return action_preds[0, -1]


def discount_cumsum(x, gamma):
    y = np.zeros_like(x)
    y[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        y[t] = x[t] + gamma * y[t + 1]
    return y


def train_step(model, get_batch, optimizer, batch_size):
    states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(batch_size)
    action_target = actions.clone()

    _, action_preds, _ = model.forward(
        states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask
    )

    act_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    loss = torch.mean((action_preds - action_target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    return loss.detach().cpu().item()


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
):
    model.eval()
    model.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0.0, 0
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            next_target = target_return[0, -1] - reward / scale
        else:
            next_target = target_return[0, -1]
        target_return = torch.cat([target_return, next_target.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1
        )

        episode_return += reward
        episode_length += 1
        if done:
            break
    return episode_return, episode_length
```

The `rtg[:, :-1]` slice matters for a mundane shape reason: the batch builder stores one extra return-to-go value, so the model consumes the first $K$ return tokens aligned with the $K$ states and actions, while the extra value remains available if I ever train a next-return head. For discrete actions and image states the changes are: a convolutional encoder in place of `embed_state`, an action lookup table instead of a linear action embedding, logits with cross-entropy instead of a tanh-regressed vector, and in the minGPT-style image code a global timestep embedding plus a local token-position embedding. The return/state/action order, the state-token action head, and the return-to-go recursion stay the same.

So the whole chain, in one breath: bootstrapping is the source of every instability in offline value learning — the deadly triad, the overestimation, the slow distractable credit assignment, the discount-induced myopia, the conservatism patches that all exist to defend against optimizing a policy against a learned value function. Drop bootstrapping entirely and the stable alternative is supervised learning, but plain cloning averages away the good behavior. Recover the good behavior by conditioning each action on the outcome — the return-to-go, the future reward still to collect — and relabeling every logged action with the return it actually achieved, which is honest supervision with no learned critic. Treat the trajectory as a token stream $(\widehat{R}_1,s_1,a_1,\dots)$ and a causal Transformer turns this into next-action prediction with full history for free, and its single-hop attention gives the model a direct way to associate distant events with returns. At test time, prompt the return I want, act, subtract each reward from the target via $\widehat{R}_{t+1}=\widehat{R}_t-r_t$, and let the model and environment keep the prefix and target consistent while useful fragments from the data can be recombined under the same supervised objective.
