I have to score one agent, one set of hyperparameters, across all 57 Atari games from raw pixels, and
read off the median human-normalized score after 200M frames. Before I can improve anything I need a floor
that *works at all* across the whole suite, because the median is unforgiving. It is the 29th of 57 sorted
scores, so it is set entirely by whether the *typical* game learns and is completely insensitive to how
spectacular the best games are: an agent that masters ten games and diverges on the other forty-seven has
forty-seven near-zero scores filling the bottom of the sort, and the 29th sits squarely inside that block,
near zero. Ten brilliant games cannot rescue it; only pushing the median-ranked game up does. So the first
question is not "how do I get a high score" but "what is the minimal recipe that learns a non-trivial policy
on a typical game without blowing up," and the honest starting point is plain Q-learning with a neural
network — whose naive version does *not* work, so I have to understand why before I can fix it.

Take the obvious thing: a convolutional network $Q(s,a;\theta)$ mapping the $84\times84\times4$ frame
stack to one value per action, trained online by Q-learning. For a transition $(s,a,r,s')$ the update
regresses $Q(s,a;\theta)$ toward $r+\gamma\max_{a'}Q(s',a';\theta)$. Run this on the live stream of an
Atari game and it oscillates or diverges, for two reasons, each pointing at a fix. The samples arrive in the
order the agent experiences them, and consecutive states are *almost the same picture*: with frame-skip 4
and a 4-frame stack, two successive stored states share three of their four frames and differ only by four
emulator frames of motion — a ball that has moved a few pixels, a score that has not changed. So successive
gradient steps see a tiny, highly correlated slice of state space, and the network overfits to whatever is
on screen right now and forgets what it learned a hundred frames ago. SGD leans hard on the i.i.d.
assumption; here the data stream violates it as badly as possible — the effective number of independent
samples in a thousand consecutive frames is closer to a handful of distinct situations than to a thousand.

The second reason is worse, and it is specific to *bootstrapping*. The target
$r+\gamma\max_{a'}Q(s',a';\theta)$ is computed from the very same $\theta$ I am updating, so the moment I
take a gradient step that raises $Q(s,a)$, I have also raised $Q(s',\cdot)$ for the many states $s'$ that
look like $s$ — which raises their targets, which pulls $Q$ up further next step. Trace the runaway
concretely: a step nudges $Q(s,a)$ up by $\eta\,\delta$ toward its target, but because the function
approximator generalizes, $\max_{a'}Q(s',a')$ for a similar $s'$ rises by some fraction $\kappa$ of that;
then the target for $s$ on the next visit has itself risen by $\gamma\kappa\,\eta\delta$, so I am chasing a
point that recedes as I approach it. If $\gamma\kappa$ is near one this is a positive feedback loop with no
fixed reference, and the values ramp until the clipping or the arithmetic saturates. And because the policy
generating the data is the greedy policy of that same moving $Q$, the *data distribution* shifts underneath
me too. Three moving parts — correlated samples, a moving target, a moving data distribution — all coupled
to one set of weights.

Before I reach for the standard cures I should ask whether something cheaper suffices, because every piece
of machinery I add is one I have to justify keeping across 57 games. The tempting minimal move is to lower
the learning rate: if the target recedes at rate $\gamma\kappa\,\eta\delta$, shrinking $\eta$ shrinks the
recession too. But this treats the wrong disease. Lowering $\eta$ slows the runaway but does nothing about
correlation — a small step on a correlated batch is still a step that overfits the current screen, and I
would need $\eta$ so small the agent learns nothing in $200$M frames. The correlation is *structural*, a
property of how the data arrives, not of how big my steps are; no learning rate fixes it. So the design has
to break the couplings at their source, and there are two sources, so I expect two pieces.

Start with the samples. The cure for correlation and for the shifting distribution is the same: stop
training on the live stream. Store every transition $(s,a,r,s',\text{done})$ in a large buffer — a replay
memory of the last $\sim$$10^6$ transitions — and each update draw a *uniform random minibatch* from it. Two
things happen at once. The minibatch mixes transitions from many different times and past policies, so the
samples in a batch are decorrelated and the effective training distribution is an average over a long
stretch of history rather than the last few frames — it changes slowly instead of lurching. And each
transition gets reused many times across its lifetime in the buffer — pure data efficiency stacked on top of
the stability. The buffer costs memory, and the arithmetic dictates *how* I store it: $10^6$ states at
$84\times84\times4$ bytes is $2.8\times10^{10}$ bytes, about $28$ GB, impractical to hold naively. But
successive states overlap by three frames, so I store single $84\times84$ frames (uint8, one byte each) and
reconstruct the 4-stack at sample time: $10^6\times84\times84\approx7\times10^9$ bytes, about $7$ GB — a
factor of four saved, and now it fits. I divide by $255$ only inside the forward pass.

The buffer size is itself a knob with a two-sided tradeoff. Over $200$M frames the agent takes $50$M
gradient-relevant steps, so a $10^6$-transition buffer holds only the most recent $2\%$ of the run. Much
smaller and the window is short enough that a minibatch re-samples nearly-adjacent transitions (correlation
creeps back) and the training distribution tracks the current policy too closely. Much larger and the buffer
fills with transitions from policies so ancient they teach the value function about states the agent will
never revisit. A buffer around $10^6$ is the balance, and since I set it once for all $57$ games I want it in
the safe middle rather than tuned to any one title.

There is a subtlety I have to respect: a replayed transition was generated by an *older* policy than the one
I am now training, so the update is **off-policy**. That is exactly the regime Q-learning is built for: its
$\max_{a'}$ target evaluates the greedy policy regardless of which policy collected the data, so off-policy
replay and the Q-learning target fit together with no extra machinery. An on-policy target — say SARSA
bootstrapping from the action actually taken — could not replay old data at all, because the stored action
was chosen by a policy I have since abandoned. The off-policy target is what *unlocks* replay; the two are
not independent design decisions.

Now the moving-target problem, which replay does not touch. The minimal fix is to give the target its own,
*slowly changing* weights. Keep a second copy $\theta^-$ of the network, the **target network**, compute the
bootstrap value $\max_{a'}Q(s',a';\theta^-)$ from it, and every $C$ updates hard-copy
$\theta^-\leftarrow\theta$. Between copies $\theta^-$ is frozen, so for that whole interval the regression
target is a *fixed* function — ordinary supervised regression toward a stationary target, which is exactly
the well-behaved problem SGD knows how to solve. Return to the runaway trace: with a frozen $\theta^-$, the
coupling coefficient $\kappa$ that fed the loop is gone, because raising $Q(s',\cdot;\theta)$ no longer moves
the target, which reads $\theta^-$. The sync period $C$ is a two-sided tradeoff again: too short (in the
limit $C=1$) and $\theta^-$ tracks $\theta$ so closely I am back to computing the target from the weights I
am updating; too long and $\theta^-$ falls far behind the improving online net, stable but slow. A period of
a few thousand steps is long enough that within any interval the target is effectively frozen yet short
enough that it is refreshed to the current agent many times over the run. The cost is one extra network's
worth of memory and one extra forward pass — cheap, and the only thing that makes the bootstrap stable.

The obvious "smoother" alternative is a soft update, $\theta^-\leftarrow(1-\tau)\theta^-+\tau\theta$ with
tiny $\tau$, tracking $\theta$ continuously instead of freezing. But that moves the target a little *every
step*, so it is never actually frozen — and it is the exactly-frozen intervals that make the sub-problem
ordinary supervised regression, the property I most want on a divergence-prone loop. It also adds a new
sensitive knob $\tau$ that would have to be right across all $57$ games, whereas a hard period $C$ in the
thousands is forgiving over a broad range. For the floor I take the hard sync: simpler, cleaner stationarity,
one fewer delicate knob to share across the suite.

A few details are not optional if one set of hyperparameters is to behave across all 57 games. Rewards in
Atari span wildly different scales — Pong's $\pm1$ per point versus Video Pinball's tens of thousands — and a
single learning rate cannot serve a squared TD error that is sometimes $0.5$ and sometimes $10^4$: the
large-reward game would produce gradients four orders of magnitude bigger and dictate the shared weights. So
clip the reward to $\{-1,0,+1\}$ during training. It throws away reward *magnitude* — the agent can no longer
tell a big score from a small one, only positive from negative — but it lets one $\gamma=0.99$, one learning
rate, one loss serve every game, which is the whole constraint of this benchmark. On the same logic, clip the
TD error (a Huber-type loss with unit threshold) so a single large error cannot produce a gradient that
wrecks the net — quadratic for $|\delta|\le1$, linear beyond, the gradient magnitude capped at $1$.
Exploration is the cheapest thing that works: $\epsilon$-greedy annealed from $1$ down to $0.1$ over the first
chunk of training and then held, so the buffer fills with varied behavior early and the policy sharpens late.
And $\gamma=0.99$ sets an effective horizon of $1/(1-\gamma)=100$ agent steps, about $400$ frames or six or
seven seconds of game time — long enough to connect an action to a delayed reward on the timescale Atari
rewards actually arrive.

The encoder shapes have to line up: input $84\times84\times4$; first conv $32$ filters $8\times8$ stride $4$
gives $\lfloor(84-8)/4\rfloor+1=20$, so $20\times20\times32$; second $64$ filters $4\times4$ stride $2$ gives
$9\times9\times64$; third $64$ filters $3\times3$ stride $1$ gives $7\times7\times64$, which flattens to
$3136$ features — that is where the $3136$ into the FC layer comes from. Then a $512$-unit FC layer and a
linear $|\mathcal A|$-head, one forward pass scoring every action, which is what makes the $\max$ and
$\arg\max$ over next-state values cheap. The bulk of the weights sits in the $3136\times512\approx1.6$M first
FC layer, an order of magnitude more than all three conv layers combined; the encoder is comparatively
lightweight and the value head is where the capacity lives. (The $84\times84$ grayscale, max-pool, frame-skip
4 and 4-frame stack are the fixed substrate; the stack is the piece that matters for value learning, since a
single frame cannot show which way the ball is moving and the Markov assumption behind Q-learning needs that
velocity visible.)

That is the floor, and I should be clear-eyed about what it is and is not. It is the smallest recipe that
makes neural Q-learning *stable* across a suite this diverse: replay to decorrelate and average the data, a
target network to freeze the bootstrap, reward/error clipping to share hyperparameters, $\epsilon$-greedy to
explore. What it is not is *good*. It carries every gap I am deliberately not fixing yet: the target uses a
single $\max$ over noisy values, which I have reason to think is biased upward; exploration is unstructured
action-space dithering with a hand-set schedule; replay samples every transition with equal weight
regardless of how much is left to learn from it; the head is a single undifferentiated stream so the state
value is smeared across the action outputs; and the value object is one scalar per action, the mean of the
return and nothing more. Each is a place the floor is deliberately crude rather than broken. So what I expect
from this floor is a real, stable agent — solving the easy games, making genuine progress on the medium ones,
and on the typical game landing clearly above random but well short of human, i.e. a median human-normalized
score below $100\%$. That number is the baseline the rest of the ladder has to beat, and the first
improvement I reach for will attack the most obviously broken piece: the upward-biased $\max$ in the target,
because it is computed the same way on every one of the $57$ games and so is the defect most likely to be
quietly holding down the median rather than a handful of tails.
