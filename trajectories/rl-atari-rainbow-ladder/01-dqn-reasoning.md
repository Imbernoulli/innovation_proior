I have to score one agent, one set of hyperparameters, across all 57 Atari games from raw pixels, and
read off the median human-normalized score after 200M frames. Before I can improve anything I need a floor
that *works at all* across the whole suite, because the median is unforgiving in a way I should make
concrete before I write a line of code. The median over 57 games is the 29th score when I sort them; it is
determined entirely by whether the *typical* game learns, and it is completely insensitive to how
spectacular the best games are. So imagine the failure mode I most have to avoid: an agent that masters ten
games — driving their normalized scores to hundreds of percent — and diverges on the other forty-seven,
leaving them pinned near random. Sort those 57 numbers and the forty-seven near-zero scores fill positions
1 through 47; the 29th, the median, sits squarely inside that block, near zero. Ten brilliant games cannot
rescue it; only pushing the *29th-ranked* game up does. That single arithmetic fact reframes the whole
problem: the first question is not "how do I get a high score" but "what is the minimal recipe that learns
a non-trivial policy on a typical game without blowing up," and the honest starting point is plain
Q-learning with a neural network — and the honest first observation is that the naive version of that does
*not* work, so I have to understand why before I can fix it.

Take the obvious thing: a convolutional network $Q(s,a;\theta)$ mapping the $84\times84\times4$ frame
stack to one value per action, trained online by Q-learning. For a transition $(s,a,r,s')$ the update
regresses $Q(s,a;\theta)$ toward $r+\gamma\max_{a'}Q(s',a';\theta)$. Run this on the live stream of an
Atari game and it oscillates or diverges. I want to be precise about the two reasons, because each one
points at a fix. The samples arrive in the order the agent experiences them, and consecutive states are
*almost the same picture*: with frame-skip 4 and a 4-frame stack, two successive stored states share three
of their four frames and differ only by four emulator frames of motion — a ball that has moved a few
pixels, a score that has not changed. So successive gradient steps see a tiny, highly correlated slice of
state space, and the network overfits to whatever is on screen right now and forgets what it learned a
hundred frames ago. Supervised learning leans hard on the i.i.d. assumption to make SGD behave; here the
data stream violates it as badly as possible — the effective number of independent samples in a thousand
consecutive frames is closer to a handful of distinct situations than to a thousand.

The second reason is worse, and it is specific to *bootstrapping*. The target
$r+\gamma\max_{a'}Q(s',a';\theta)$ is computed from the very same $\theta$ I am updating, so the moment I
take a gradient step that raises $Q(s,a)$, I have also raised $Q(s',\cdot)$ for the many states $s'$ that
look like $s$ — which raises their targets, which pulls $Q$ up further next step. Let me trace the runaway
concretely: suppose a step nudges $Q(s,a)$ up by $\eta\,\delta$ toward its target, but because the
function approximator generalizes, $\max_{a'}Q(s',a')$ for a similar $s'$ rises by some fraction $\kappa$
of that; then the target for $s$ on the next visit has itself risen by $\gamma\kappa\,\eta\delta$, so I am
chasing a point that recedes as I approach it. If $\gamma\kappa$ is near one this is a positive feedback
loop with no fixed reference, and the values ramp until the clipping or the arithmetic saturates. And
because the policy that generates the data is the greedy policy of that same moving $Q$, the *data
distribution* shifts underneath me too. Three moving parts — correlated samples, a moving target, a moving
data distribution — all coupled to one set of weights.

Before I reach for the standard cures I should ask whether something cheaper suffices, because every piece
of machinery I add is a piece I have to justify keeping across 57 games. The tempting minimal move is to
just lower the learning rate: if the target recedes at rate $\gamma\kappa\,\eta\delta$, shrinking $\eta$
shrinks the recession too. But this treats a symptom of the wrong disease. Lowering $\eta$ slows the
runaway, but it does nothing about correlation — a small step on a correlated batch is still a step in a
direction that overfits the current screen, just a smaller one, and I would need $\eta$ so small that the
agent learns nothing in $200$M frames. The correlation is *structural*, a property of how the data arrives,
not of how big my steps are; no learning rate fixes it. So the design has to break the couplings at their
source, and there are two sources, so I expect two pieces.

Start with the samples. The cure for correlation and for the shifting distribution is the same: stop
training on the live stream. Store every transition $(s,a,r,s',\text{done})$ in a large buffer — a replay
memory of the last $\sim$$10^6$ transitions — and each update draw a *uniform random minibatch* from it.
Two things happen at once. The minibatch mixes transitions from many different times and many different
past policies, so the samples in a batch are decorrelated and the effective training distribution is an
average over a long stretch of the agent's history rather than the last few frames — it changes slowly
instead of lurching. And each transition gets reused many times across its lifetime in the buffer: at one
gradient update per four emulator frames and a buffer of $10^6$ transitions, a transition sits in memory
for on the order of a million updates before it is overwritten, so it contributes to many minibatches
rather than one — pure data efficiency stacked on top of the stability. I do have to pay for a buffer this
size, and the arithmetic is a reason to be careful about *how* I store it: $10^6$ states at
$84\times84\times4$ bytes is $2.8\times10^{10}$ bytes, about $28$ GB, which is impractical to hold in
memory naively. But successive states overlap by three frames, so I store single $84\times84$ frames
(uint8, one byte each) and reconstruct the 4-stack at sample time: $10^6\times84\times84\approx7\times10^9$
bytes, about $7$ GB — a factor of four saved, and now it fits. Keeping the frames as uint8 rather than
float is the other half of that budget; I divide by $255$ only inside the forward pass.

The size of the buffer is itself a design choice with a tradeoff I should reason through rather than
inherit. Over the full budget of $200$M frames the agent takes $50$M gradient-relevant steps
($200\text{M}/4$), so a $10^6$-transition buffer holds only the most recent $2\%$ of the run — a sliding
window of roughly the last two percent of the agent's experience. Make it much smaller and two problems
return: the window is short enough that a minibatch drawn from it re-samples nearly-adjacent transitions,
so correlation creeps back, and the training distribution tracks the current policy too closely, undoing
the averaging I wanted. Make it much larger and the buffer fills with transitions from policies so ancient
and so much worse than the current one that they teach the value function about states the agent will never
revisit — off-policy is fine, but off-policy against a policy from a hundred million frames ago is mostly
wasted capacity. A buffer around $10^6$ is the balance: long enough to decorrelate and average, short
enough that its oldest transitions are still recognizably from the same learning process. This is a knob I
am setting once, for all $57$ games, so I want it in the safe middle rather than tuned to any one title.

There is a subtlety I have to respect: a replayed transition was generated by an *older* policy than the
one I am now training, so the update is **off-policy** — I am learning about the greedy policy from data
collected by a different (older, more exploratory) policy. That is exactly the regime Q-learning is built
for: its $\max_{a'}$ target evaluates the greedy policy regardless of which policy collected the data, so
off-policy replay and the Q-learning target fit together with no extra machinery. If I had used an
on-policy target — say a SARSA update that bootstraps from the action actually taken — I could not replay
old data at all, because the action in a stale transition was chosen by a policy I have since abandoned.
The choice of an off-policy target is what *unlocks* replay; the two are not independent design decisions.

Now the moving-target problem, which replay does not touch — the target is still computed from $\theta$.
The minimal fix is to give the target its own, *slowly changing* set of weights. Keep a second copy
$\theta^-$ of the network, the **target network**, and compute the bootstrap value
$\max_{a'}Q(s',a';\theta^-)$ from it; every $C$ updates (a few thousand steps), hard-copy
$\theta^-\leftarrow\theta$. Between copies $\theta^-$ is frozen, so for that whole interval the regression
target is a *fixed* function — I am doing ordinary supervised regression toward a stationary target, which
is exactly the well-behaved problem SGD knows how to solve. Return to the runaway trace: with a frozen
$\theta^-$, the coupling coefficient $\kappa$ that fed the loop is gone, because raising $Q(s',\cdot;\theta)$
no longer moves the target, which reads $\theta^-$; the recession rate drops to zero for the length of the
sync interval. The sync period $C$ is the same kind of two-sided tradeoff as the buffer size, and worth pinning down. Make
$C$ too short — sync every step, in the limit $C=1$ — and $\theta^-$ tracks $\theta$ so closely that I am
back to computing the target from the weights I am updating, and the runaway loop I just traced returns.
Make $C$ too long and $\theta^-$ falls far behind the improving online net, so I spend a long interval
regressing toward the estimates of an agent I have already surpassed, which is stable but slow — the
bootstrap carries stale information backward. A period of a few thousand steps is long enough that within
any interval the target is effectively frozen (thousands of gradient steps against a fixed function, which
is exactly the regime SGD is happy in) yet short enough that the target is refreshed to the current agent
many times over the run. The periodic refresh keeps the target from falling too far behind the improving
online net, so I get the stability of a fixed target without the staleness of a permanently frozen one. The
cost is one extra network's worth of memory and one extra forward pass on the next states — cheap, and the
only thing that makes the bootstrap stable. This is the difference between chasing a runaway target and
stepping toward a target that holds still long enough to reach.

There is an alternative to the hard copy worth weighing, because it is the obvious "smoother" thing:
instead of freezing $\theta^-$ and hard-copying every $C$ steps, track $\theta$ continuously with a slow
exponential average, $\theta^-\leftarrow(1-\tau)\theta^-+\tau\theta$ with a tiny $\tau$. This never lets
the target jump, which sounds gentler than the discontinuous hard sync. But think about what it does to the
stationarity I am buying: with a soft update the target moves a little *every single step*, so it is never
actually frozen — the regression target is always drifting, just slowly. The hard-sync scheme instead gives
me genuine intervals of exact stationarity punctuated by a jump, and it is the *exactly-frozen* intervals
that make the sub-problem ordinary supervised regression, which is the property I most want on a
divergence-prone loop. A soft update also introduces a new sensitive hyperparameter $\tau$ that would have
to be right across all $57$ games, whereas a hard period $C$ in the thousands is forgiving — anywhere in a
broad range gives long-enough frozen intervals. So for the floor I take the hard sync: it is simpler, it
has the cleaner stationarity story, and it has one fewer delicate knob to share across the suite.

A few details that are not optional if I want this to behave across all 57 games with *one* set of
hyperparameters. Rewards in Atari span wildly different scales — Pong's $\pm1$ per point versus Q*bert's
hundreds and Video Pinball's tens of thousands — and a single learning rate cannot serve a squared TD error
that is sometimes $0.5$ and sometimes $10^4$: the game with the large rewards would produce gradients four
orders of magnitude bigger and simply dictate the shared weights, drowning every small-reward game. So clip
the reward to $\{-1,0,+1\}$ during training. It throws away reward *magnitude* — the agent can no longer
tell a big score from a small one, only positive from negative — but it lets one $\gamma=0.99$, one learning
rate, one loss serve every game, which is the whole constraint of this benchmark. On the same logic, clip
the TD error (equivalently, use a Huber-type loss with unit threshold) so a single large error cannot
produce a gradient that wrecks the net — quadratic for $|\delta|\le1$, linear beyond, so the gradient
magnitude is capped at $1$ no matter how wrong a single bootstrap is. Exploration is the cheapest possible
thing that still works: $\epsilon$-greedy with $\epsilon$ annealed from $1$ down to $0.1$ over the first
chunk of training — roughly the first $20$M frames, which at four frames per step is about $5$M agent steps
— and then held fixed, so the buffer fills with varied behavior early and the policy sharpens late. And
$\gamma=0.99$ is not arbitrary: it sets an effective horizon of $1/(1-\gamma)=100$ agent steps, which at
four frames per step and sixty frames per second is about $400$ frames or roughly six or seven seconds of
game time — long enough to connect an action to a delayed reward on the timescale Atari rewards actually
arrive.

The preprocessing that feeds this encoder is part of the fixed substrate, but it is worth understanding why
each piece is there, because it shapes what the value function can even see. RGB is collapsed to grayscale
and downsampled to $84\times84$ to cut the input by a large factor with negligible loss for these games —
color rarely carries decision-relevant information that luminance does not, and $84\times84$ is small enough
to convolve cheaply yet large enough to resolve the sprites. Two consecutive emulator frames are max-pooled
before downsampling because the 2600 hardware flickers sprites on alternating frames to fit its object
limit; without the max-pool an object can vanish from the state on the frame it happened to be skipped, and
the agent would be learning from a partially blind observation. Frame-skip $4$ repeats each chosen action
for four emulator frames, which quarters the decision rate — the agent acts on the timescale things
actually change rather than every $1/60$ s — and the $4$-frame stack restores the *motion* information a
single frame throws away: from one $84\times84$ image I cannot tell which way the ball is moving, but from
four stacked frames the velocity is visible, which the Markov assumption behind Q-learning quietly requires.
So the state the encoder receives is already engineered to be approximately Markov and approximately
flicker-free; that is a precondition for the whole value-learning story and not something the network is
expected to discover.

The encoder is worth deriving rather than quoting, because the shapes have to line up or nothing runs.
Input is $84\times84\times4$. First conv is $32$ filters of $8\times8$ at stride $4$:
$\lfloor(84-8)/4\rfloor+1=20$, giving $20\times20\times32$. Second is $64$ filters of $4\times4$ at stride
$2$: $\lfloor(20-4)/2\rfloor+1=9$, giving $9\times9\times64$. Third is $64$ filters of $3\times3$ at stride
$1$: $\lfloor(9-3)/1\rfloor+1=7$, giving $7\times7\times64$, which flattens to $7\times7\times64=3136$
features — that is where the $3136$ into the FC layer comes from, not a magic constant. Then a $512$-unit
fully-connected layer and a linear head of $|\mathcal A|$ outputs, one forward pass scoring every action,
which is what makes the $\max$ and the $\arg\max$ over next-state values cheap. Counting parameters tells me
where the model's capacity actually lives: conv1 has $32\times(4\times8\times8)=8192$ weights, conv2
$64\times(32\times4\times4)=32768$, conv3 $64\times(64\times3\times3)=36864$, but the first FC layer alone
is $3136\times512\approx1.6$M — an order of magnitude more than all three conv layers combined. So the bulk
of the weights sit in the transition from convolutional features to the value head, which is exactly the
part later rungs will want to restructure; the encoder is comparatively lightweight.

Let me sanity-check the whole update once by tracing the simplest non-trivial case, because if the pieces
do not compose on a two-state example they will not compose on Pong. Take a terminal transition,
$\text{done}=1$: the target collapses to $y=r$, a constant, so the update is pure supervised regression of
$Q(s,a)$ onto the observed reward — no bootstrap, no instability, and clipping makes $|r|\le1$ so the
gradient is bounded. Now a one-step-from-terminal transition, $\text{done}=0$ into a state $s'$ all of whose
actions lead to termination with reward $0$: the frozen target is $y=r+\gamma\cdot0=r$, again a constant for
the whole sync interval, and $Q(s,a)$ regresses toward it without chasing. Only when I chain many such
states does the bootstrap carry information backward, and the frozen $\theta^-$ guarantees it carries it
toward a target that is not simultaneously moving. The dimensions check too: $Q(\text{obs})$ is
$(B,|\mathcal A|)$, gathering the taken action gives $(B,)$, the target net's max over next-state values is
$(B,)$, and the Huber loss reduces $(B,)$ to a scalar — the shapes close, and the $(1-\text{done})$ mask is
a $(B,)$ vector that zeroes exactly the transitions where there is no next state. So the loop is well-posed
at the boundary cases where instability would first show up.

That is the floor, and I should be clear-eyed about what it is and is not. It is the smallest recipe that
makes neural Q-learning *stable* across a suite this diverse: replay to decorrelate and average the data, a
target network to freeze the bootstrap, reward/error clipping to share hyperparameters, $\epsilon$-greedy to
explore. What it is not is *good*. It carries every gap the prior art left open and I am deliberately not
fixing yet: the target still uses a single $\max$ over noisy values, which I have every reason to think is
biased upward; exploration is unstructured action-space dithering with a hand-set schedule; replay samples
every transition with equal weight regardless of how much is left to learn from it; the head is a single
undifferentiated stream so the state value is smeared across the action outputs; and the value object is one
scalar per action, the mean of the return and nothing more. Each of those is a place the floor is
deliberately crude rather than broken, and cataloguing them now is exactly how I know what a later
change would even be measured against — the point of pinning this floor first is to have that number. So what I expect from this rung is a real, stable agent — solving the
easy games, making genuine progress on the medium ones, and on the typical ("median") game landing somewhere
clearly above random but well short of human, i.e. a median human-normalized score below $100\%$. That
median number is the baseline the rest of the ladder has to beat, and the first improvement I reach for will
be the one that attacks the most obviously broken piece of this floor: the upward-biased $\max$ in the
target, because it is computed the same way on every one of the $57$ games and so is the defect most likely
to be quietly holding down the median rather than a handful of tails.
