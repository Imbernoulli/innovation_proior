Prioritizing replay was the broad win I hoped for — 140% median, the biggest step since the decoupled
target. I have now fixed the target's bias, the exploration, and how the buffer samples; the loss and the
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
$1/|\mathcal A|$ as often as it should be and only ever through the lens of one action. That is a structural
mismatch between what I need ($V$ everywhere) and how the architecture learns it (through sparse per-action
updates). This is not an algorithm bug — the target is fine — it is the *shape* of the function
approximator fighting the structure of the problem.

So reshape the head, not the algorithm. Split the 512 features into two streams: a **value stream**
producing a scalar $V(s)$, and an **advantage stream** producing a vector $A(s,a)$, then recombine them
into the single $Q$ output the rest of the agent expects, so replay, the target, $\epsilon$-noise — all of
it — sees the identical state$\to Q$ interface and nothing else has to change. The point of splitting is
that now $V(s)$ sits *underneath all $|\mathcal A|$ outputs*: every transition, whatever action it used,
backpropagates into the value stream, so the shared state value is learned from *all* the data instead of a
$1/|\mathcal A|$ slice. And the advantage stream is free to drive the many redundant actions toward
$A\approx0$ rather than independently fitting each one's value — it only has to represent the *differences*,
which is genuinely low-dimensional in the states where actions are interchangeable.

The naive recombination $Q=V+A$ has a problem I have to fix or the split is meaningless: it is
*unidentifiable*. Add a constant $c$ to $V$ and subtract $c$ from every $A(s,a)$ and $Q$ is unchanged, so
gradient descent has no pressure to make the value stream learn the *actual* $V$ — it could put any
arbitrary offset there and compensate in the advantages, and then $V$ is not the state value at all, it is
junk plus a constant, and I have gained nothing. I need to *pin the offset*: subtract a per-state reference
from the advantage before adding it to $V$, so the two streams can no longer trade a free constant. Two
choices for the reference. The clean-semantics one subtracts the max:
$Q(s,a)=V(s)+\big(A(s,a)-\max_{a'}A(s,a')\big)$; then at the greedy action the bracket is zero so
$Q(s,a^\star)=V(s)$ and $V$ is *exactly* $\max_a Q$, which is what I want it to mean. But $\max_{a'}A$ jumps
discontinuously whenever the best action flips, and that jump destabilizes training. The practical choice
subtracts the *mean*: $Q(s,a)=V(s)+\big(A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')\big)$. This gives up the
exact "$V=\max Q$" semantics — $V$ becomes the value plus the mean advantage offset, the centered advantage
becomes $Q(s,a)-\operatorname{mean}_{a'}Q(s,a')$ — but the mean is a *smooth* reference that does not jump
when the argmax flips, so the advantage stream only has to track a slowly-moving baseline rather than
compensate every change in the best action. More stable, and the cost is only a reinterpretation of what
$V$ absorbs, not a loss of the identifiability fix.

The crucial invariant: subtracting a per-state *constant* (max or mean, same for all actions in that state)
never changes the *rank order* of the actions, so $\arg\max_a Q(s,a)$ is identical to the naive sum's — the
greedy and $\epsilon$-greedy policies are exactly what they would have been. The aggregator is purely a
training-time offset-control device; it changes how the streams are *learned*, not the policy they
parameterize. That is what makes this a clean architectural rung: same policy class, same algorithm, only
the head's internal structure and its gradient flow differ.

A couple of stability details that the split forces. Both streams backprop into the *shared* conv trunk, so
the trunk now receives roughly twice the incoming gradient it did with one stream; rescale the gradient
entering the trunk by $1/\sqrt2$ to keep its magnitude in the range the encoder was stable at. And clip the
global gradient norm (to $\le10$), with a slightly lower learning rate than plain Double DQN, because the
two-stream head is a little more delicate early in training. The streams are each a 512-unit FC layer +
ReLU off the shared trunk; value stream $\to1$, advantage stream $\to|\mathcal A|$; combine with the
mean aggregator inside the forward pass. I keep the Double-DQN target underneath (this composes with it
unchanged) — the head reshape is orthogonal to how the target is built.

Now the bar. This should help on the broad middle of the suite for a reason that is general rather than
game-specific: *every* Atari game has many states where the action is nearly irrelevant, so every game
benefits from learning $V$ from all transitions instead of a $1/|\mathcal A|$ slice — and the benefit is
larger the *larger* the action set, because that is where the value dilution under the single-stream head
was worst. So I expect another broad lift, clearing 140%, of the same character as the prioritized-replay
gain rather than the tail-only noisy-nets one. What it cannot fix — and I am noting it because it is the
last rung's job — is that $Q$ (and now $V$ and $A$) are still *scalars*, point estimates of a return whose
whole distribution the agent never sees. The head is now well-shaped, but it is shaped to predict a mean.
The remaining gap is the value object itself.
