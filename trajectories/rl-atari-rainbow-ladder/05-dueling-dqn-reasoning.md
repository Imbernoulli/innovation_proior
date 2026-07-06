Prioritizing replay was the broad win I hoped for — 140% median, up $22$ points from the $118\%$ noisy-nets
baseline, the biggest step since the decoupled target's $38$. That confirms the diagnosis: the typical game
was bottlenecked on data efficiency, on spending its gradient budget re-learning transitions it already
predicted, and reallocating that budget toward the surprising transitions lifted the broad middle exactly
as a median-moving change should. Both of my broad wins so far ($+38$, $+22$) came from fixing something the
floor did on *every* game — the biased max, the wasteful uniform sampling — while the one narrow result
($+1$) came from fixing something only a minority needed. That pattern is my compass now: to move the median
again I want another defect the floor commits on every game, not a minority's problem. I have now fixed the
target's bias, the exploration, and how the buffer samples; the loss and the
discount are reasonable. Two axes are still untouched: the *architecture of the head* and the *value object
itself* (still a scalar). The scalar object is a deeper, representational change and I want to take it last,
so the cheaper structural move comes first: the head. Right now the head is a single undifferentiated
stream — the 512 features feed one linear map to $|\mathcal A|$ outputs, and each action's value is
estimated independently. Is that the right shape for the thing I am learning? Let me look at what a typical
Atari state actually demands of it.

In a large fraction of Atari states the *action barely matters*. The ball is mid-flight and nowhere near
the paddle; the screen is between events; whatever I do for the next few frames, the outcome is essentially
the same. In those states the action values are all nearly equal, and the only quantity that carries
information is their common level — *how good is it to be here at all*, the state value $V(s)$. The action
choice matters only in the sparse, decisive states (the ball is about to reach the paddle; an enemy is in
range). So the object I am learning is really two things glued together: a state value $V(s)$ that matters
*everywhere*, and a per-action *advantage* $A(s,a)=Q(s,a)-V(s)$ that matters only in the decisive states
and is near zero elsewhere.

Now look at how the single-stream head learns $V(s)$. In a TD update on a transition $(s,a,\dots)$, only the
$a$-th output gets a gradient — I regress $Q(s,a)$ toward its target and the other actions' outputs are
untouched. The state value is not represented as its own quantity; it is implicit, smeared across all
$|\mathcal A|$ outputs, and it only gets corrected through whichever single action happened to be sampled.
So the one quantity a bootstrapping algorithm needs accurate at *every* state — because every target is
$r+\gamma V(s')$-shaped through the next-state value — is the most *diluted* thing in the network, updated
$1/|\mathcal A|$ as often as it should be and only ever through the lens of one action. Put the action
counts in: on a game with the full $18$-action set, the single-stream head updates the value information
carried by any one action's output only when *that* action is the one sampled, so each output — and the
state-value content smeared into it — receives a gradient roughly $1/18$ of the transitions, an
$18{\times}$ dilution of exactly the quantity I most need everywhere. Even a modest six-action game dilutes
it sixfold. And it is worse than a simple rate, because the state value is not cleanly localized in any one
output — it is entangled across all $|\mathcal A|$ of them, so a gradient on action $a$'s output has to
re-derive the shared $V$ from that one action's target, and a gradient on action $b$ does it again
independently, with no shared parameter forcing them to agree on the common level. The architecture makes
the network learn one number ($V$) $|\mathcal A|$ separate times from $|\mathcal A|$ disjoint slices of the
data. That is a structural
mismatch between what I need ($V$ everywhere) and how the architecture learns it (through sparse per-action
updates). This is not an algorithm bug — the target is fine — it is the *shape* of the function
approximator fighting the structure of the problem.

What are my options for fixing this, and why is a two-stream split the right one? The laziest move is to
just make the single-stream head bigger — more units, another layer — but that does not touch the problem:
extra capacity still learns the state value through per-action outputs updated $1/|\mathcal A|$ of the time,
so I would be adding parameters to a structure that dilutes $V$ by construction. Capacity is not the
bottleneck; the *shape* is. A second option is to keep the single head but add an auxiliary loss that
explicitly regresses a separate $V(s)$ head toward the bootstrap value — but "the bootstrap value" is
itself $\max_a Q$, so I would be inventing a target for $V$ out of the very quantity I am trying to learn,
and I would have to hand-weight that auxiliary loss against the main one, another shared hyperparameter to
get right across $57$ games. The clean option is to make $V$ a *structural* part of the forward pass so it
receives gradient on every transition automatically, with no extra loss and no extra target: split the head.

So reshape the head, not the algorithm. Split the 512 features into two streams: a **value stream**
producing a scalar $V(s)$, and an **advantage stream** producing a vector $A(s,a)$, then recombine them
into the single $Q$ output the rest of the agent expects, so replay, the target, $\epsilon$-noise — all of
it — sees the identical state$\to Q$ interface and nothing else has to change. The point of splitting is
that now $V(s)$ sits *underneath all $|\mathcal A|$ outputs*: every transition, whatever action it used,
backpropagates into the value stream, so the shared state value is learned from *all* the data instead of a
$1/|\mathcal A|$ slice. This lands exactly where the bootstrap needs it. Every target the agent regresses
toward is $r+\gamma\,($value of $s')$-shaped, so the accuracy of the *next-state value* is what governs the
quality of every update on every transition; a well-learned $V$ feeds cleaner targets back into the whole
buffer. So the change is not cosmetic re-labeling — it moves the gradient that used to reach one action's
output to reach a parameter ($V$) that every target depends on, which is why I expect it to help broadly
rather than on any special class of game. And the advantage stream is free to drive the many redundant actions toward
$A\approx0$ rather than independently fitting each one's value — it only has to represent the *differences*,
which is genuinely low-dimensional in the states where actions are interchangeable. Count what each shape
has to represent in an action-irrelevant state. The single-stream head must output $|\mathcal A|$ numbers
that all happen to be nearly equal to the common value $V$ — it has to *coordinate* $|\mathcal A|$ separate
outputs to agree, with no parameter enforcing the agreement, so any per-action noise shows up as spurious
advantage. The dueling head outputs one $V$ plus $|\mathcal A|$ advantages that the aggregator wants near
zero; the "everything is equal" fact is expressed by a single scalar ($V$) plus a near-zero vector, which
is a far easier target for the approximator to hit than $|\mathcal A|$ independently-fit equal numbers. So
in the common case — the majority of Atari states — the dueling parameterization is not just fed more
gradient into $V$, it is representing the state in genuinely fewer effective degrees of freedom, which is
why it should both learn faster and be less noisy in exactly the states that dominate the data.

The naive recombination $Q=V+A$ has a problem I have to fix or the split is meaningless: it is
*unidentifiable*. Verify it directly — under $V\to V+c$ and $A(s,a)\to A(s,a)-c$ for every action,
$Q'=(V+c)+(A-c)=V+A=Q$ exactly, for any state-dependent $c$. So $Q$ is completely blind to the split; the
loss, which only ever sees $Q$, therefore has no pressure to make the value stream learn the *actual* $V$ — it could put any
arbitrary offset there and compensate in the advantages, and then $V$ is not the state value at all, it is
junk plus a constant, and I have gained nothing. I need to *pin the offset*: subtract a per-state reference
from the advantage before adding it to $V$, so the two streams can no longer trade a free constant. Two
choices for the reference. The clean-semantics one subtracts the max:
$Q(s,a)=V(s)+\big(A(s,a)-\max_{a'}A(s,a')\big)$; then at the greedy action the bracket is zero so
$Q(s,a^\star)=V(s)$ and $V$ is *exactly* $\max_a Q$, which is what I want it to mean. But $\max_{a'}A$ jumps
discontinuously whenever the best action flips, and that jump destabilizes training. Make the jump concrete:
suppose two actions have advantages hovering near each other, $A_1=0.50$ and $A_2=0.49$, so
$\max_{a'}A=0.50$ and the reference subtracted from $V$ is $0.50$. A tiny gradient step nudges them to
$A_1=0.49$, $A_2=0.51$; now the reference is $0.51$, so $V$'s effective offset has jumped by $0.01$
*discontinuously* even though nothing meaningful changed — the identity of the argmax flipped, and the
reference is a non-smooth function of the advantages precisely at that crossover. Near any state where two
actions compete, the max reference chatters, and the value stream is forced to chase a target that steps
around every time the lead changes hands. The mean has no such crossover: it is a smooth average of all
advantages, so an infinitesimal change in the advantages moves the reference infinitesimally. The practical choice
subtracts the *mean*: $Q(s,a)=V(s)+\big(A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')\big)$. Check that the
mean reference actually breaks the degeneracy: under $V\to V+c$, $A\to A-c$, the new output is
$(V+c)+\big((A-c)-\frac1{|\mathcal A|}\sum_{a'}(A-c)\big)=(V+c)+\big(A-c-\operatorname{mean}(A)+c\big)
=V+A-\operatorname{mean}(A)+c$, which differs from the original by exactly $c$ — so the free constant no
longer cancels, and gradient descent now *does* feel pressure on where it puts the offset. The value stream
can no longer be junk-plus-a-constant compensated in the advantages, because any such shift changes $Q$ and
hence the loss. This gives up the exact "$V=\max Q$" semantics — $V$ becomes the value plus the mean advantage offset, the centered advantage
becomes $Q(s,a)-\operatorname{mean}_{a'}Q(s,a')$ — but the mean is a *smooth* reference that does not jump
when the argmax flips, so the advantage stream only has to track a slowly-moving baseline rather than
compensate every change in the best action. More stable, and the cost is only a reinterpretation of what
$V$ absorbs, not a loss of the identifiability fix. Trace one state to see what the mean version actually
outputs. Say the value stream emits $V=5.0$ and the advantage stream emits $A=(1.0,-0.4,0.0)$ for three
actions; the mean advantage is $(1.0-0.4+0.0)/3=0.2$, so the centered advantages are $(0.8,-0.6,-0.2)$ and
$Q=(5.8,4.4,4.8)$. The three $Q$ values average to $5.0$, so $V$ has become exactly the mean of the action
values rather than their max — that is the "reinterpretation" — and the action ranking $1\succ3\succ2$ is
read straight off the advantage differences, unchanged by the reference. If the advantage stream drifts,
the mean tracks it smoothly and $V$ re-centers gently; there is no crossover where $Q$ leaps. The example
also shows the identifiability fix at work: I cannot add a constant to $V$ and absorb it in $A$ without
changing these three $Q$ numbers, so the split is pinned.

The crucial invariant: subtracting a per-state *constant* (max or mean, same for all actions in that state)
never changes the *rank order* of the actions, so $\arg\max_a Q(s,a)$ is identical to the naive sum's — the
greedy and $\epsilon$-greedy policies are exactly what they would have been. Verify with a pairwise
comparison: $Q(s,a)-Q(s,b)=\big(V+A(s,a)-\kappa(s)\big)-\big(V+A(s,b)-\kappa(s)\big)=A(s,a)-A(s,b)$, where
$\kappa(s)$ is the per-state reference — the shared $V$ and the shared $\kappa(s)$ cancel in every pairwise
difference. Since all pairwise orderings are set by the advantage differences alone and are untouched by
$V$ or by the reference subtraction, the $\arg\max$, the full ranking, and hence the policy are exactly
preserved. The aggregator is purely a
training-time offset-control device; it changes how the streams are *learned*, not the policy they
parameterize. That is what makes this a clean architectural rung: same policy class, same algorithm, only
the head's internal structure and its gradient flow differ.

A couple of stability details that the split forces, and the first one has a clean derivation rather than a
fudge. Both streams backprop into the *shared* conv trunk, so the trunk now receives the *sum* of two
gradient contributions where before it received one. If those two contributions are roughly independent and
each of order $g$, their sum has variance $\approx2g^2$, so its typical magnitude is $\sqrt2\,g$ — the
trunk suddenly sees gradients about $40\%$ larger than the single-stream encoder was tuned for, which can
push it out of the stable regime. Rescale the gradient entering the trunk by $1/\sqrt2$ and the magnitude
returns to $g$: $\frac{1}{\sqrt2}\cdot\sqrt2\,g=g$. The $1/\sqrt2$ is not a tuned constant, it is exactly
the factor that undoes the variance-doubling of summing two streams, which is why it transfers across all
$57$ games without per-game adjustment. On top of that I count the parameter cost: the single-stream head
was one $3136\to512$ layer plus a $512\to|\mathcal A|$ map; the dueling head is *two* $3136\to512$ layers
(one per stream) plus a $512\to1$ value map and a $512\to|\mathcal A|$ advantage map. The dominant added
cost is the second $3136\times512\approx1.6$M-parameter layer — essentially one extra copy of the head's big
matrix, on a $\sim1.7$M-parameter network. That is a real increase, but it buys a structural change that
lets $V$ be learned from all transitions, which the parameters alone could never do in the single-stream
shape. And clip the
global gradient norm (to $\le10$), with a slightly lower learning rate than plain Double DQN, because the
two-stream head is a little more delicate early in training: before the mean aggregator has settled the
offset, the value and advantage streams can briefly push against each other — one drifting up while the
other compensates down — and produce a large transient gradient, so a norm clip that caps the whole update
at $10$ keeps one such transient from knocking the shared trunk off its stable path. This is a different
instrument from the $1/\sqrt2$ trunk rescale: the rescale corrects the *expected* doubling of gradient
variance that happens on every step, while the norm clip catches the *occasional* large excursion, so I want
both — one for a systematic scale shift, one for a tail risk. The streams are each a 512-unit FC layer +
ReLU off the shared trunk; value stream $\to1$, advantage stream $\to|\mathcal A|$; combine with the
mean aggregator inside the forward pass. I keep the Double-DQN target underneath (this composes with it
unchanged) — the head reshape is orthogonal to how the target is built, because the aggregator only changes
how the streams produce the scalar $Q$, and the decoupled target only changes which network's $\arg\max$
selects the bootstrap action; neither touches the other's machinery, so I can hold the target fixed and
attribute any median change to the head alone. That orthogonality is what lets this be a clean single-axis
rung on top of the established target rather than a two-variable change.

Now the bar. This should help on the broad middle of the suite for a reason that is general rather than
game-specific: *every* Atari game has many states where the action is nearly irrelevant, so every game
benefits from learning $V$ from all transitions instead of a $1/|\mathcal A|$ slice — and the benefit is
larger the *larger* the action set, because that is where the value dilution under the single-stream head
was worst. So I expect another broad lift, clearing 140%, of the same character as the prioritized-replay
gain rather than the tail-only noisy-nets one — and, like the overestimation fix, larger where the action
count is larger, since that is where the single-stream dilution was worst ($18{\times}$ on the full-action
games versus sixfold on the small ones). That gives me a falsifiable prediction with a shape, not just a
direction: not only should the median rise above $140\%$, but the per-game improvement should correlate with
action-set size, biggest on the many-action games where $V$ was most starved. If instead the median barely
moves, or the gains show up on small-action games where the dilution argument says they should not, then the
value-dilution story is wrong and the head was not the bottleneck. What it cannot fix — and I have been
deliberately deferring this since the start — is that $Q$ (and now $V$ and $A$) are still *scalars*, point
estimates, one number apiece. Every axis I have changed so far has
been a *recipe* tweak: a better target, a better sampling rule, a better exploration mechanism, a
better-shaped head. All of them still learn the same object, the mean of the return, just more accurately or
more efficiently. The head is now well-shaped, but it is shaped to predict a single number, and that is still true after all
five changes: the object the network predicts is a scalar, the mean of the return, exactly as it was on the
floor. That the learned object is still one number per action is the one thing none of these recipe
improvements has altered — a representational fact about the current agent, of a different kind than the
structural tweaks I have been making, and I note it here as the property this rung leaves exactly where it
found it.
