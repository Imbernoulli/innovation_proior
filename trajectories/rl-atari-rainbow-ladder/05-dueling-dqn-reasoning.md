Prioritizing replay was the broad win I hoped for — 140% median, up $22$ points from the $118\%$ noisy-nets
baseline, the biggest step since the decoupled target's $38$. That confirms the diagnosis: the typical game
was bottlenecked on data efficiency, on spending its gradient budget re-learning transitions it already
predicted. Both broad wins so far ($+38$, $+22$) came from fixing something the floor did on *every* game —
the biased max, the wasteful uniform sampling — while the one narrow result ($+1$) fixed something only a
minority needed. That pattern is my compass: to move the median again I want another defect the floor commits
on every game. I have fixed the target's bias, the exploration, and how the buffer samples; the loss and
discount are reasonable. Two axes remain: the *architecture of the head* and the *value object* itself. The
scalar object is a deeper, representational change I want last, so the cheaper structural move comes first.
Right now the head is a single undifferentiated stream — the 512 features feed one linear map to
$|\mathcal A|$ outputs, each action's value estimated independently. Is that the right shape? Look at what a
typical Atari state demands.

In a large fraction of Atari states the *action barely matters*: the ball is mid-flight and nowhere near the
paddle, the screen is between events, whatever I do for the next few frames the outcome is essentially the
same. In those states the action values are all nearly equal, and the only quantity carrying information is
their common level — *how good is it to be here at all*, the state value $V(s)$. The action choice matters
only in the sparse decisive states. So the object I am learning is really two things glued together: a state
value $V(s)$ that matters *everywhere*, and a per-action *advantage* $A(s,a)=Q(s,a)-V(s)$ that matters only
in the decisive states and is near zero elsewhere.

Now look at how the single-stream head learns $V(s)$. In a TD update on $(s,a,\dots)$, only the $a$-th output
gets a gradient — the other actions' outputs are untouched. The state value is not represented as its own
quantity; it is implicit, smeared across all $|\mathcal A|$ outputs, and corrected only through whichever
action was sampled. So the one quantity a bootstrapping algorithm needs accurate at *every* state — every
target is $r+\gamma V(s')$-shaped through the next-state value — is the most *diluted* thing in the network,
updated $1/|\mathcal A|$ as often as it should be. On the full $18$-action set each output receives a
gradient roughly $1/18$ of transitions, an $18{\times}$ dilution of exactly the quantity I most need
everywhere; even a modest six-action game dilutes it sixfold. And it is worse than a rate, because $V$ is not
cleanly localized in any output — it is entangled across all of them, so a gradient on action $a$ has to
re-derive the shared $V$ from that one action's target and a gradient on action $b$ does it again
independently, with no parameter forcing them to agree. The architecture makes the network learn one number
($V$) $|\mathcal A|$ separate times from $|\mathcal A|$ disjoint slices of the data. This is not an algorithm
bug — the target is fine — it is the *shape* of the function approximator fighting the problem.

The options: making the single-stream head bigger does not touch it — extra capacity still learns $V$ through
per-action outputs updated $1/|\mathcal A|$ of the time; capacity is not the bottleneck, the shape is. Adding
an auxiliary loss that regresses a separate $V$ head toward the bootstrap value fails because "the bootstrap
value" is itself $\max_a Q$, so I would be inventing a target for $V$ out of the quantity I am trying to
learn, plus a hand-weighted auxiliary loss to tune across $57$ games. The clean option is to make $V$ a
*structural* part of the forward pass so it receives gradient on every transition automatically, with no
extra loss and no extra target: split the head.

So reshape the head, not the algorithm. Split the 512 features into a **value stream** producing a scalar
$V(s)$ and an **advantage stream** producing a vector $A(s,a)$, then recombine into the single $Q$ output the
rest of the agent expects — replay, the target, the noise all see the identical state$\to Q$ interface.
Now $V(s)$ sits *underneath all $|\mathcal A|$ outputs*: every transition, whatever action it used,
backpropagates into the value stream, so the shared state value is learned from *all* the data instead of a
$1/|\mathcal A|$ slice. This lands where the bootstrap needs it — every target is $r+\gamma\,($value of
$s')$-shaped, so the accuracy of the next-state value governs the quality of every update, and a well-learned
$V$ feeds cleaner targets back into the whole buffer. And the advantage stream only has to represent the
*differences*, which is genuinely low-dimensional where actions are interchangeable: in an action-irrelevant
state the single-stream head must *coordinate* $|\mathcal A|$ outputs to agree with no parameter enforcing
it, so any per-action noise shows up as spurious advantage, whereas the dueling head expresses "everything is
equal" as one scalar $V$ plus a near-zero vector — far easier to hit. So in the common case the dueling
parameterization is not just fed more gradient into $V$, it represents the state in fewer effective degrees
of freedom, which is why it should both learn faster and be less noisy where the data concentrates.

The naive recombination $Q=V+A$ is *unidentifiable*: under $V\to V+c$ and $A(s,a)\to A(s,a)-c$ for every
action, $Q'=(V+c)+(A-c)=V+A=Q$ exactly, for any state-dependent $c$. So $Q$ is blind to the split; the loss,
which only sees $Q$, has no pressure to make the value stream learn the *actual* $V$ — it could put any
offset there and compensate in the advantages, and then $V$ is junk plus a constant. I need to *pin the
offset*: subtract a per-state reference from the advantage before adding it to $V$. The clean-semantics
choice subtracts the max, $Q(s,a)=V(s)+\big(A(s,a)-\max_{a'}A(s,a')\big)$; then at the greedy action the
bracket is zero, $Q(s,a^\star)=V(s)$, and $V$ is *exactly* $\max_a Q$. But $\max_{a'}A$ jumps discontinuously
whenever the best action flips: with $A_1=0.50$, $A_2=0.49$ the reference is $0.50$; a tiny step to
$A_1=0.49$, $A_2=0.51$ makes it $0.51$, so $V$'s effective offset jumps by $0.01$ even though nothing
meaningful changed, and near any state where two actions compete the reference chatters. The **mean** has no
crossover: $Q(s,a)=V(s)+\big(A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')\big)$. It still breaks the
degeneracy — under $V\to V+c$, $A\to A-c$ the output becomes $V+A-\operatorname{mean}(A)+c$, differing from
the original by exactly $c$, so the free constant no longer cancels and gradient descent now feels pressure
on where it puts the offset. This gives up exact "$V=\max Q$" semantics ($V$ becomes value plus the mean
advantage; the centered advantage becomes $Q(s,a)-\operatorname{mean}_{a'}Q(s,a')$), but the mean is a
smooth reference the advantage stream can track without compensating every change in the best action — more
stable, at the cost of only a reinterpretation of what $V$ absorbs.

The crucial invariant: subtracting a per-state *constant* (max or mean, same for all actions in that state)
never changes the *rank order* of the actions, so the greedy and $\epsilon$-greedy policies are exactly what
they would have been under the naive sum. Pairwise,
$Q(s,a)-Q(s,b)=\big(V+A(s,a)-\kappa(s)\big)-\big(V+A(s,b)-\kappa(s)\big)=A(s,a)-A(s,b)$: the shared $V$ and
the shared reference $\kappa(s)$ cancel in every pairwise difference, so all orderings are set by the
advantage differences alone. The aggregator is purely a training-time offset-control device; it changes how
the streams are *learned*, not the policy they parameterize — same policy class, same algorithm, only the
head's internal structure differs.

A couple of stability details the split forces. Both streams backprop into the *shared* conv trunk, so the
trunk now receives the *sum* of two gradient contributions where before it received one; if each is order $g$
and roughly independent, their sum has variance $\approx2g^2$, magnitude $\sqrt2\,g$ — about $40\%$ larger
than the single-stream encoder was tuned for. Rescale the gradient entering the trunk by $1/\sqrt2$ and the
magnitude returns to $g$. The $1/\sqrt2$ is not a tuned constant, it is exactly the factor that undoes the
variance-doubling of summing two streams, which is why it transfers across all $57$ games. The parameter cost
is one extra $3136\times512\approx1.6$M layer — the second stream's big matrix — on a $\sim1.7$M-parameter
network, a real increase that buys a structural change capacity alone could not. On top I clip the global
gradient norm ($\le10$) with a slightly lower learning rate, because early on, before the mean aggregator has
settled the offset, the value and advantage streams can briefly push against each other and produce a large
transient gradient; the norm clip catches that *occasional* excursion while the $1/\sqrt2$ rescale corrects
the *expected* per-step scale shift, so I want both. I keep the Double-DQN target underneath: the aggregator
only changes how the streams produce the scalar $Q$, and the decoupled target only changes which network's
$\arg\max$ selects the bootstrap action, so the two are orthogonal and any median change is attributable to
the head alone.

Now the bar. This should help on the broad middle for a general reason: *every* Atari game has many states
where the action is nearly irrelevant, so every game benefits from learning $V$ from all transitions instead
of a $1/|\mathcal A|$ slice — and the benefit is larger the *larger* the action set, since that is where the
dilution was worst ($18{\times}$ on full-action games versus sixfold on small ones). So I expect another
broad lift, clearing 140%, of the same character as the prioritized-replay gain rather than the tail-only
noisy-nets one, and with a *shape*: the per-game improvement should correlate with action-set size, biggest
on the many-action games where $V$ was most starved. If instead the median barely moves, or the gains show up
on small-action games where the dilution argument says they should not, the value-dilution story is wrong.
What this cannot fix — and I have deferred it since the start — is that $Q$, $V$, and $A$ are still *scalars*,
point estimates of the mean return. Every axis I have changed has been a *recipe* tweak — a better target,
sampling rule, exploration mechanism, head shape — all still learning the same object, the mean of the
return, more accurately or efficiently. The head is now well-shaped, but shaped to predict a single number,
and that is the one property none of these changes has altered, of a different kind than the structural
tweaks — which is exactly the lever I have left for last.
