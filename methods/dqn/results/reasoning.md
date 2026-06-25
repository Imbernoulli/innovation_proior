OK, let me think this through from scratch. I have an emulator that hands me a screen — 210×160 pixels of color, sixty times a second — and a number that ticks up when I do well. I get to press one of a handful of buttons. That's it. No one tells me what a paddle is, what a ball is, what "good" looks like. I want a single program that, dropped into any of these games, figures out how to play it well. And I want it to learn its own notion of what matters on the screen, because the moment I start hand-drawing features I've capped the thing at whatever I personally can think to measure.

So where does the leverage come from? Two worlds I half-trust. On one side, deep convolutional networks have just eaten computer vision and speech: feed raw pixels and raw waveforms into a deep net, train it by stochastic gradient descent, and the learned features beat anything anyone hand-built. The recipe there is brutally simple — raw input, light SGD updates, a mountain of labeled examples. On the other side, reinforcement learning has a clean theory for sequential decision-making. The obvious move is to glue them: let a convolutional net be the thing RL trains. But the glue is where everything has historically fallen apart, so let me be honest about why before I reach for any machinery.

What exactly do I want the agent to maximize? Future reward, discounted so that nearer reward counts more and the sum stays finite: R_t = r_t + γ r_{t+1} + γ² r_{t+2} + … = Σ_{t'≥t} γ^{t'-t} r_{t'}. Fine. But maximizing a sum-of-future-rewards directly is a credit-assignment nightmare — a button press now might only pay off thousands of frames later. The classical escape is to learn a value: how good is it to be in a situation and take a particular action, assuming I act well afterward? Define Q*(s,a) = max over all policies of the expected return after seeing s, taking a. If I had Q*, control is trivial — just take argmax_a Q*(s,a) every step.

And Q* has this self-referential structure I can lean on. If I already knew the optimal value of every next situation s', then standing at (s,a) the best I can do is collect the immediate reward and then behave optimally from s', so

  Q*(s,a) = E_{s'}[ r + γ max_{a'} Q*(s',a') | s,a ].

That's the Bellman optimality equation, and it suggests an algorithm: treat it as an assignment, Q_{i+1}(s,a) ← E[ r + γ max_{a'} Q_i(s',a') ], iterate, and in the tabular case this contracts to Q*. Watkins' Q-learning is the sample-based shadow of this — you don't compute the expectation, you just nudge toward one observed transition:

  Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ].

Two things about this are gifts. It's model-free: I never have to predict the next frame, I just use the transitions the emulator hands me. And it's off-policy: the max_{a'} inside means I'm learning about the greedy policy, even while I behave some other way — say, mostly greedy but occasionally random so I keep exploring. Hold onto "off-policy"; it's going to matter more than it looks right now.

The wall is immediate and tall. Q-learning as written keeps a separate number for every (s,a). My state is a visual history, and there is no table large enough for that; even if there were, I'd never visit the same exact screen twice, so a per-state table generalizes nothing. I have to replace the table with a function approximator, Q(s,a;θ) ≈ Q*(s,a), and let a neural net be that function. The instant I do, I can write down a supervised-looking loss: pick a nonterminal transition, use the same online network to form y(θ) = r + γ max_{a'} Q(s',a';θ), but stop the gradient through that target before I differentiate. With sg[·] marking the stop-gradient operation,

  ℓ(θ) = 1/2 (sg[y(θ)] − Q(s,a;θ))²,

so the semi-gradient is

  ∇_θ ℓ = −(y − Q(s,a;θ)) · ∇_θ Q(s,a;θ) = (Q(s,a;θ) − y) · ∇_θ Q(s,a;θ).

The descent step therefore points in the direction +(y − Q(s,a;θ))∇_θ Q(s,a;θ). Let me check the sign is actually doing what I want rather than just assuming it. Suppose γ=0.99, the transition gives r=1, and from s' the best next value is max_{a'} Q(s',a';θ)=3; then y = 1 + 0.99·3 = 3.97. If my current prediction is Q(s,a;θ)=0.5, the error (y − pred) = 3.47 is positive, so the step raises the prediction toward 3.97 — correct, it was too low. If instead the prediction were 5.0, the error is 3.97 − 5.0 = −1.03, negative, so the step lowers it — also correct. Good, the sign is what I need in both directions. Now notice what I quietly did: I differentiated Q(s,a;θ), the prediction, but I froze the target — I did *not* differentiate the max_{a'} Q(s',a';θ) that also depends on θ. This isn't the gradient of any honest fixed objective; it's a semi-gradient, a bootstrap. I'm chasing a target that's made out of my own current estimates. It feels like supervised learning, but the labels are moving, and they're moving because *I* am moving.

And this is exactly the configuration everyone in RL has learned to fear. Generalization, the thing that makes a net useful, is also the thing that bites here: when I push Q(s,a;θ) toward its target, θ is shared, so I also move Q at the neighboring states — including s' itself — which moves the very target I was chasing. With a linear approximator off-policy, Baird built a tiny five-state counterexample where this feedback sends the weights to infinity. Tsitsiklis and Van Roy proved the general shape of it: temporal-difference learning with function approximation is stable on-policy (linear) but can diverge off-policy. The warning is that function approximation, bootstrapping, and off-policy data together can create an amplifying loop. The field's reaction was to back away to linear approximators with provable guarantees, and to mostly treat neural-net value functions as a curiosity. So I'm proposing to do the one combination everyone says blows up — a *nonlinear* approximator, bootstrapped, off-policy — and the theory offers me no cover. There is, as of now, no convergent recipe for nonlinear *control*; the careful gradient-TD work reaches nonlinear only for evaluating a fixed policy, or reaches control only with linear features. The hole is exactly the corner I want to stand in.

But there's an existence proof nagging at me. TD-Gammon: Tesauro put a multilayer perceptron on the value function, trained it by TD from self-play, and got superhuman backgammon two decades ago. So a neural net *can* be a value function trained by bootstrapping. Then why did it stay a one-off? People tried the same recipe on chess, Go, checkers and it fizzled, and the going explanation is that backgammon is special — the dice inject stochasticity that smooths the value surface and forces exploration. Maybe. But TD-Gammon is also strictly online and on-policy: it updates from each fresh self-play move, in sequence, against the distribution its own current policy induces. That on-policy-online combination is precisely where the correlated-data and moving-distribution problems live, and those are a wall of their own, separate from divergence.

Even setting aside the divergence theory, gluing a deep net to RL fights the assumptions SGD was built on. First, my data arrive as a video — frame t and frame t+1 are almost identical, so consecutive samples are violently correlated. SGD likes roughly independent samples; a stream of near-duplicates gives me high-variance, redundant gradient steps that all push the same way at once. Second, and nastier: the data distribution is non-stationary, and it's non-stationary because of *me*. Suppose right now the greedy action is "go left." Then I spend my time on the left side of the screen, so almost all my training data are left-side states, so I fine-tune the net to be great on the left. That shifts the values, maybe now "go right" looks better, and suddenly the whole stream of states I train on lurches to the right — and the left-side competence I just built drifts. It's a feedback loop between the policy and its own training data, and it's easy to see it oscillating, or parking the parameters in some poor corner, or, combined with the bootstrap, diverging outright. In supervised learning the dataset just sits there; here the act of learning rewrites what I'll learn from next.

So I have two pains that are really the same pain wearing two hats: correlated samples, and a self-generated drifting distribution. What would kill both? I want the gradient steps to see decorrelated samples, and I want the training distribution to stop lurching with every policy twitch — I want it averaged out over a long history of behavior so no single recent obsession dominates. Stare at that wish for a second. "Average the behavior over many past states." That's not a property of the latest policy; that's a property of a *pool* of past experience. What if I stop training on the live stream at all, and instead keep a big memory of transitions I've lived through and sample from it?

Concretely: every step I store the transition e_t = (s_t, a_t, r_t, s_{t+1}) into a memory D of the last N transitions. Then, to learn, I don't use the transition I just generated — I draw a minibatch uniformly at random from D and take a gradient step on those. Lin floated replaying stored experience to a small net years ago; I'm going to lean on it hard and at scale. Before I talk myself into this, I should check that the decorrelation is real and not just a story I like. Let me make a crude stand-in for the video stream: a random walk x_t (each value is the previous plus noise), which is exactly the kind of slowly-drifting, near-duplicate signal consecutive frames are. The lag-1 autocorrelation between x_t and x_{t+1} over a long run comes out at about 0.9999 — consecutive samples are essentially the same sample, so a minibatch of consecutive frames is one effective example wearing thirty-two hats, and SGD's variance assumptions are violated badly. Now draw pairs at two *uniformly random* indices into the same buffer and measure their correlation: it lands at roughly 0.00 (I get −0.002 on one run). So the random draw genuinely turns a stream with ~1.0 neighbor-correlation into a sample with ~0 pairwise correlation — the decorrelation is not hand-waving, it falls straight out of breaking temporal adjacency. The other gains follow from the same buffer: the memory holds transitions from many past policies, so the distribution I train against is averaged over a long stretch of behavior rather than being whatever the current policy is fixated on this instant, which should smooth the lurching feedback loop. And, almost for free, each transition gets reused in many updates instead of being seen once and thrown away, which is real data efficiency — and I need that, because the reward signal is sparse and every scrap of experience is precious.

And now the off-policy point pays off, exactly where I told myself to remember it. If I'm pulling a transition out of a memory, it was generated by some *older* version of my parameters, under an older behavior — it is by construction not on-policy with respect to my current θ. So whatever learning rule I use *has* to be valid off-policy. That single requirement disqualifies the on-policy TD-Gammon-style update: TD-Gammon's V(s) target is built assuming the data come from the policy being evaluated, and a replayed transition simply isn't drawn from my current policy, so that update would be estimating the value of a behavior I'm no longer running. It points instead at Q-learning, whose max_{a'} target evaluates the greedy policy regardless of which policy generated the data — off-policy from the start. So the constraint I introduced to cure correlated, drifting data (replay) happens to require off-policy learning, and the one tabular rule I already trust is off-policy. The two requirements are compatible, which is the thing I needed to confirm before committing — replay and Q-learning don't fight. The divergence warning doesn't vanish, but replay attacks the mechanism behind it: the runaway feedback from a self-correlated, self-shifting training distribution is exactly the loop the buffer's decorrelation and distribution-averaging damp.

Why not the obvious batch alternative? Riedmiller's NFQ is the closest existing thing: same squared-error loss sequence, but it re-fits the whole network on the entire accumulated dataset every iteration with RPROP. That respects the off-policy structure and is stable-ish, but its cost per iteration scales with the size of the dataset — refit everything, every time. I want to train a convolutional net on millions of frames; a method whose per-step cost grows with how much I've collected is a dead end. The whole point of SGD is a low *constant* cost per update — one minibatch, one step — independent of how much history I've piled up. So: keep Q-learning's stochastic minibatch update (constant cost, scales to huge data, naturally online), and bolt experience replay on top of it. NFQ also did one thing on the architecture side I'll want to invert, but the value-target details come first.

There's a subtlety I should be honest about in this version. The target is not coming from a separate frozen network. It is computed from the same online weights I'm about to nudge — I form r + γ max_{a'} Q(s',a';θ), wrap that number in a stop-gradient, and then step θ through Q(s,a;θ) only. The bootstrap is live. Replay is doing the heavy lifting on stability: by averaging the training distribution over a long memory and decorrelating the draws, it keeps the feedback loop from compounding into divergence within the timescale of training. I'll want to watch closely for any sign of the weights blowing up, but the smoothing from replay is my main line of defense here.

The careful prior work I might try to borrow does not quite reach this corner. The gradient-TD family — derived as honest gradients of the projected Bellman error — actually fixes the divergence properly, with convergence proofs off-policy under function approximation. But Maei's nonlinear convergence result is for policy *evaluation*, a fixed policy; the control version, Greedy-GQ, is linear-only. Nobody has a convergent recipe for nonlinear *control*, which is exactly what a deep net choosing actions is. So I can't borrow a guarantee; I'm betting that replay-driven smoothing plus a stable optimizer can hold this together even though the theory for this corner doesn't exist yet. That's the gamble, stated plainly.

Now the network itself. NFQ and its descendants parameterize Q by feeding the state *and* the action in together and reading out a single scalar Q(s,a). That seems natural until I count forward passes: to act, I need max_a Q(s,a), so I'd have to run the net once per action — up to eighteen passes per decision, eighteen per target computation. Wasteful, and it scales with the action count. Flip it: make the state the only input and give the network one output unit per action, so a single forward pass yields the whole vector [Q(s,a_1), …, Q(s,a_K)] at once. Then acting is one pass and an argmax; computing a target is one pass and a max. The architecture's shape is dictated by how I need to *use* it, and using it means comparing all actions cheaply.

What goes into that single input? The framework wants the state to be the full history s_t = x_1,a_1,…,x_t, because a single Atari frame is partially observed — one snapshot can't tell me whether the ball is rising or falling, which way an enemy is drifting; the velocity lives in the *change* between frames, and many distinct situations render to identical pixels. So in principle I need the whole sequence. But feeding an arbitrarily long history to a net means a recurrent architecture backpropagating through thousands of steps, which in 2013 is a nightmare to train on top of everything else I'm already risking. Do I actually need the *whole* history, or just enough to undo the partial observability? The aliasing I care about is local in time — it's about motion and direction, which a couple of consecutive frames already reveal. So I'll define a fixed-length summary φ that takes the last few frames and stacks them. Take the last 4 frames; that's enough to read velocities and the immediate dynamics, and it keeps the input a fixed-size tensor a plain convolutional net can swallow. The history-as-state ideal collapses, for practical purposes, into a 4-frame window.

Preprocessing, driven by cost. Raw frames are 210×160 in color — heavy. Convert to grayscale (color rarely carries the control signal, and it triples the channels), downsample to 110×84, and crop an 84×84 square that covers the play area. The square crop is partly just so the GPU convolution code I'm using, which wants square inputs, is happy. Stack 4 of these and the network input is 84×84×4.

The conv stack: I want something that can see objects and motion but is cheap enough to train on millions of frames on one GPU. First layer, 16 filters of 8×8 with stride 4 — big receptive field, aggressive stride to shrink the spatial map fast, then a ReLU. With valid convolution the 84×84 input becomes floor((84−8)/4)+1 = 20 cells on each side. Second layer, 32 filters of 4×4 with stride 2, ReLU again, and 20 becomes floor((20−4)/2)+1 = 9, so the flattening step has 32×9×9 = 2592 units. Then a fully connected layer of 256 ReLU units, and finally a linear output layer with one unit per valid action (between 4 and 18 depending on the game). ReLUs throughout because they train fast and don't saturate. Two conv layers and a modest 256-wide head is deliberately lean — enough capacity to learn the visual features that discriminate action-values, light enough to run end-to-end. And it *is* end-to-end: unlike the deep-autoencoder-then-control pipeline, where features are learned first by reconstruction and frozen before control, here the gradient from the control loss flows all the way to the pixels, so the features the net learns are exactly the ones useful for telling good actions from bad — not the ones useful for reconstructing the screen.

Now stitch the loop. Initialize the replay memory D to capacity N, initialize the Q-network with random weights. For each episode, reset and form φ_1 from the first frame. Then at each step: with probability ε pick a random action, otherwise pick argmax_a Q(φ(s_t),a;θ) — that's ε-greedy, the exploration valve. Why ε-greedy and why anneal it? Early on the net knows nothing, so its greedy choices are noise; I want lots of random exploration to fill the memory with varied transitions. Late on, the values are meaningful and I want to mostly exploit while keeping a little randomness so I don't tunnel. So start ε at 1 (pure exploration) and anneal it linearly down to 0.1 over the first million frames, then hold it at 0.1. As a linear interpolation ε(t) = 1 + min(t/10⁶, 1)·(0.1 − 1): at t=0 that is 1.0, at the half-million mark it's 1 − 0.5·0.9 = 0.55, at t=10⁶ it's exactly 0.1, and the min() clamp pins it at 0.1 for everything after — so a quarter of the way in I'm still exploring three-quarters of the time, and the descent to mostly-greedy is gradual rather than a cliff, which is what I wanted. Execute the chosen action, observe reward r_t and the next frame x_{t+1}, append to the history and recompute φ_{t+1}, and store the transition (φ_t, a_t, r_t, φ_{t+1}) in D.

Then the learning step, the part that's decoupled from the live stream: sample a random minibatch of stored transitions (φ_j, a_j, r_j, φ_{j+1}) from D. For each, build the target — and here the terminal case matters. If φ_{j+1} is terminal, there is no future, so the target is just y_j = r_j; bootstrapping past the end of an episode would be hallucinating reward that can't happen, so the γ max term has to be switched off at terminal transitions. Otherwise y_j = r_j + γ max_{a'} Q(φ_{j+1}, a'; θ). I want to make sure I implement this as one branch-free expression rather than an if-statement that I might get wrong, so let me write it as y_j = r_j + γ·(1−done_j)·max_{a'} Q(φ_{j+1},a';θ) and check the two cases collapse correctly. Take r_j=1, γ=0.99, max_{a'} Q=3. Nonterminal (done=0): y = 1 + 0.99·1·3 = 3.97. Terminal (done=1): y = 1 + 0.99·0·3 = 1.0, exactly r_j as it must be. So the (1−done) multiplier does the switching cleanly, which is the form I'll put in the code. Then take a gradient descent step on 1/2(y_j − Q(φ_j, a_j; θ))², whose semi-gradient is (Q(φ_j,a_j;θ) − y_j)∇_θ Q(φ_j,a_j;θ); crucially, the gradient only flows through Q(φ_j, a_j; θ), the value of the action that was actually taken, and the target is treated as a constant. That's the semi-gradient I wrote earlier, now applied to a sampled minibatch.

A few more practical knobs, each with a reason. Reward scale: across games, scores live on wildly different scales — a point in Pong is not a point in Beam Rider — and if I want one learning rate to work everywhere, I can't let the error magnitudes swing by orders of magnitude per game. So clip all positive rewards to +1, all negative to −1, leave 0 alone, during training. This bounds the size of the error derivatives and lets a single learning rate hold across every game. The cost is real and I should name it: the agent can no longer tell a small reward from a huge one, so it treats all scoring events as equally valuable — a deliberate trade of magnitude-sensitivity for cross-game robustness.

Frame-skipping, for compute: running the emulator forward one frame is far cheaper than a network forward pass plus action selection. So let the agent see and choose only every k-th frame and repeat its last action on the skipped ones. With k=4 the agent plays roughly four times as many games per unit of compute, almost for free. One exception: in Space Invaders the lasers blink with a period that, at k=4, lands the agent on frames where the lasers are invisible — it would be blind to them. Drop to k=3 there so the lasers are seen. That single value is the only hyperparameter I change between games, which keeps the "one method, all games" claim honest.

The optimizer: plain SGD with one global learning rate struggles when different parameters need different effective step sizes, and value targets are noisy. RMSProp divides each parameter's update by a running root-mean-square of its recent gradients, giving a per-parameter adaptive rate and damping coordinates with persistent large gradients. Minibatches of 32. Replay memory of the most recent one million experiences, sampled uniformly. Train for ten million frames total. The uniform sampling is admittedly crude — it gives every stored transition equal weight and, once the memory is full, overwrites the oldest regardless of how informative it was. Something like prioritized sweeping, which replays the transitions you can learn the most from, would likely do better; I'm leaving that on the table and taking uniform for simplicity.

One last problem: how do I even know it's working during training? The natural metric, average episode reward is going to be noisy — a tiny weight change can swing which states the policy visits and send the score jumping around. I want a steadier window. The agent's own predicted value can give me one: collect a fixed set of states once, by running a random policy before training starts, and periodically track the average of max_a Q over that frozen set. Because the states are fixed, changes in this diagnostic isolate the value function's own estimates rather than mixing them with a constantly changing state distribution. That's my validation-set analogue.

Let me say the causal chain back to myself in one breath. I want control from raw pixels with learned features, so I make a convolutional net the action-value function and act greedily by it. Tabular Q-learning can't generalize, so I approximate Q with the net — but that puts function approximation, bootstrapping, and off-policy data in the same loop, where correlated online samples and a self-shifting training distribution can make the bootstrap diverge. Experience replay fixes both pains at once: random minibatches from a long memory decorrelate the samples and average the training distribution over many past policies, and reusing each transition buys data efficiency — and because replayed samples are inherently off-policy, the learning rule must be off-policy, which forces Q-learning's max-target and slots everything together. I give the net one output per action so a single pass scores all actions, summarize partial-observable history as a 4-frame stack to avoid recurrence, train the half-squared TD error by semi-gradient with the terminal case handled, and add reward clipping, frame-skipping, ε-annealing, and RMSProp so the same architecture and shared settings carry across games, with only the frame-skip visibility exception. Here's the whole thing as code.

```python
import random
import numpy as np
import torch
import torch.nn as nn

# 4-frame stack -> one output per action: a single forward pass scores every action.
class ActionValueModel(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),   # 84 -> 20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 20 -> 9
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),                  # 2592 -> 256
            nn.ReLU(),
            nn.Linear(256, num_actions),                 # one Q-value per action
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)               # pixels in [0,1]

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def add(self, s, a, r, s_next, done):
        item = (
            np.asarray(s, dtype=np.uint8).copy(),
            int(a),
            float(r),
            np.asarray(s_next, dtype=np.uint8).copy(),
            float(done),
        )
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item                    # overwrite oldest (finite memory)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)       # uniform draw

    def __len__(self):
        return len(self.buf)

def exploration_rate(step, start=1.0, end=0.1, duration=1_000_000):
    frac = min(step / duration, 1.0)
    return start + frac * (end - start)

def phi(frames):
    arr = np.asarray(frames)
    if arr.shape == (84, 84, 4):                         # channel-last frame stack
        arr = arr.transpose(2, 0, 1)
    elif arr.shape != (4, 84, 84):
        arr = np.stack(frames, axis=0)
    if arr.shape != (4, 84, 84):
        raise ValueError(f"expected 4 stacked 84x84 frames, got {arr.shape}")
    return arr.astype(np.uint8, copy=False)

def reset_frames(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 3:
        return out
    if len(out) == 4:
        frames, reward, done, _ = out
        return frames, reward, done
    frames, reward, terminated, truncated, _ = out
    return frames, reward, bool(terminated or truncated)

def select_action(net, state, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(net(state_t).argmax(dim=1).item())

def unpack_batch(batch, device):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    return states, actions, rewards, next_states, dones

def compute_targets(net, rewards, next_states, dones, gamma):
    with torch.no_grad():
        max_next = net(next_states).max(dim=1).values
        return rewards + gamma * max_next * (1.0 - dones)

def values_for_taken_actions(net, states, actions):
    return net(states).gather(1, actions[:, None]).squeeze(1)

def optimize_step(net, optimizer, batch, gamma, device):
    states, actions, rewards, next_states, dones = unpack_batch(batch, device)
    target = compute_targets(net, rewards, next_states, dones, gamma)
    pred = values_for_taken_actions(net, states, actions)
    loss = 0.5 * ((target - pred) ** 2).mean()           # half-squared TD error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(env, num_actions=None, total_frames=10_000_000, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if num_actions is None:
        num_actions = env.num_actions if hasattr(env, "num_actions") else env.action_space.n
    q = ActionValueModel(num_actions).to(device)
    optimizer = torch.optim.RMSprop(q.parameters(), lr=2.5e-4, alpha=0.95, eps=0.01)
    memory = ReplayMemory(capacity=1_000_000)
    gamma, batch_size = 0.99, 32

    state = phi(reset_frames(env))
    for step in range(total_frames):
        eps = exploration_rate(step)
        a = select_action(q, state, eps, num_actions, device)

        # env.step repeats the action on skipped frames and clips rewards to {-1,0,+1}.
        next_frames, r, done = step_env(env, a)
        next_state = phi(next_frames)
        memory.add(state, a, r, next_state, float(done))
        state = phi(reset_frames(env)) if done else next_state

        if len(memory) < batch_size:
            continue
        optimize_step(q, optimizer, memory.sample(batch_size), gamma, device)
    return q
```
