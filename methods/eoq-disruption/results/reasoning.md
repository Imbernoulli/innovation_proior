OK, let me think this through from scratch. I have one item, steady demand $D$, one supplier, and I want to know how much to order each time. If the supplier were always there I'd be done already — that's Harris 1913. Deplete from $Q$ down to zero at rate $D$, refill instantly, sawtooth between $Q$ and $0$, average inventory $Q/2$, and the cost per unit time is the ordering rate plus the holding: $FD/Q + aD + hQ/2$. Differentiate, the $FD/Q$ and $hQ/2$ trade off, and out drops $Q_E=\sqrt{2FD/h}$. Zero-inventory ordering is optimal there: never hold a unit longer than you must, because the instant you run dry you can refill. Clean.

But my supplier isn't always there. It goes down — fire, strike, flood — for random stretches, and while it's down it ships nothing. So picture the sawtooth again: I deplete to zero, I go to reorder, and… the supplier is down. Now I just sit. No inventory, demand still arriving, and I either lose those sales or backlog them, until the supplier comes back and I can finally place the order. The clean deterministic cycle is gone. The length of a cycle is now a random variable, because how long I wait at the bottom depends on whether the supplier happens to be up when I hit zero, and if not, how long until it recovers.

So the first thing that breaks is the cost accounting itself. In Harris every cycle is identical — length $Q/D$ — so "cost per cycle over cycle length" and "long-run cost per unit time" are the same trivial thing. The moment cycle length is random I can't just divide by $Q/D$ anymore. What's the right way to average over random cycles? This is exactly what renewal-reward is for: if the process regenerates — comes back to a clean, memoryless fresh start at random epochs — then the long-run average cost per unit time is the expected cost incurred in one cycle divided by the expected length of one cycle,
$$ I = \frac{E[\text{cost per cycle}]}{E[\text{cycle length}]}, $$
even though both are random. The classical case is the degenerate version where the denominator is the constant $Q/D$. Good — so I haven't thrown anything away, I've just promoted the denominator from a constant to an expectation. Now I need two things: $E[\text{cycle length}]$ and $E[\text{cost per cycle}]$, both as functions of $Q$.

For "cycle" I'll take the duration between two consecutive *successful* shipments — i.e. between the moments I actually get an order in. That's the natural regeneration point: right after a shipment arrives I'm holding exactly $Q$ and the supplier is, by definition, up (it just shipped to me). So each cycle starts in an identical state — inventory $Q$, supplier up — and because the supplier's up-time is memoryless, the future from that point doesn't depend on the past. It's a genuine renewal.

Now the cycle length. It's the depletion time $Q/D$ — the time to burn through $Q$ at rate $D$ — plus however long I wait at the bottom if the supplier happens to be down when I get there. So I need: when I hit zero (which is $Q/D$ after the cycle started, and the supplier was up at the start), what's the chance the supplier is now down, and if it is, how long do I expect to wait?

Let me get the supplier model precise. Up for an exponential time with rate $\lambda$, then down for an exponential time with rate $\psi$, alternating — a two-state continuous-time Markov chain. $\lambda$ is how fast it fails (disruption rate), $\psi$ is how fast it recovers (recovery rate). I want the probability that, having started in the up state, the chain is in the down state $t$ later. Call it $\phi(t)$. Set up the forward equation: let $p(t)$ be the probability of being down at time $t$, starting up so $p(0)=0$. Down-probability gains from up-states failing and loses from down-states recovering:
$$ p'(t) = \lambda(1-p(t)) - \psi\,p(t) = \lambda - (\lambda+\psi)p(t). $$
The steady state is $p_\infty = \lambda/(\lambda+\psi)$, and the homogeneous part decays at rate $\lambda+\psi$, so with $p(0)=0$,
$$ \phi(t) = \frac{\lambda}{\lambda+\psi}\left(1-e^{-(\lambda+\psi)t}\right). $$
Sanity: $\phi(0)=0$ (it starts up), and $\phi(\infty)=\lambda/(\lambda+\psi)$, the long-run fraction of time down. Right.

So when I reach zero, $t=Q/D$ has elapsed, and the probability the supplier is down then is $\phi(Q/D)$. If it's up — refill instantly, no wait. If it's down — I wait until it recovers. How long is that wait, in expectation? Here's where memorylessness pays off: the down-time is exponential with rate $\psi$, and exponentials forget how long they've already run, so regardless of when during the down period I arrived, my expected remaining wait is just the mean down-time $1/\psi$. So the expected extra wait at the bottom of a cycle is $\phi(Q/D)\cdot\frac{1}{\psi}$.

Let me assemble $E[T]$:
$$ E[T] = \frac{Q}{D} + \frac{1}{\psi}\,\phi\!\left(\frac{Q}{D}\right) = \frac{Q}{D} + \frac{1}{\psi}\cdot\frac{\lambda}{\lambda+\psi}\left(1-e^{-(\lambda+\psi)Q/D}\right). $$
The combination $\frac{1}{\psi}\cdot\frac{\lambda}{\lambda+\psi} = \frac{\lambda}{\psi(\lambda+\psi)}$ keeps showing up; let me name it $A_0 = \frac{\lambda}{\psi(\lambda+\psi)}$. Then
$$ E[T] = \frac{Q}{D} + A_0\left(1-e^{-(\lambda+\psi)Q/D}\right). $$
Notice the structure: the depletion time $Q/D$ plus a *disruption surcharge* on the cycle. And notice the limit — if $\lambda\to 0$, the supplier never fails, $A_0\to 0$, and $E[T]\to Q/D$. The classical cycle length falls right back out. That's the reassurance I want at every step: turn off disruptions and I'm back to Harris.

Now the cost per cycle. Three pieces. Ordering: every cycle I place one order of size $Q$, fixed plus variable, so $F + aQ$. (Lost sales, so I really do order exactly $Q$ each time.) Holding: the inventory still runs the classical triangle from $Q$ down to $0$ over $Q/D$ — demand is deterministic and I'm using zero-inventory ordering, so on-hand traces the same straight line down as in Harris, regardless of how long I then wait at the bottom holding nothing. The area under that triangle is $\frac{1}{2}Q\cdot\frac{Q}{D} = \frac{Q^2}{2D}$, so holding cost per cycle is $h\frac{Q^2}{2D}$. The waiting at the bottom adds zero holding cost because I'm holding zero.

The third piece is the shortage cost, and this is where I have to be careful — it's the piece that's genuinely new. During the wait at the bottom of the cycle I have no stock, demand keeps arriving at rate $D$, and each unit I can't serve costs me $\pi$ (a lost sale). How much demand goes unserved per cycle? Exactly the demand that arrives during the time I'm at zero with the supplier down. The expected length of the whole cycle is $E[T]$; of that, the expected time I actually *have* inventory is $Q/D$ (I'm depleting the triangle). So the expected time with no inventory is $E[T]-Q/D$, the unmet demand is $D(E[T]-Q/D)$, and the shortage cost per cycle is
$$ \pi D\left(E[T]-\frac{Q}{D}\right). $$
Let me pause on this, because there's a subtle trap here that I want to make sure I'm not falling into. A naive version of this model would say "every time I hit zero and the supplier is down, I take a stockout, charge for it." But that double-thinks the situation in two ways. First, a stockout only happens if the supplier is *actually* down when I arrive — not on every cycle — and $\phi(Q/D)$ already weighs exactly that probability into $E[T]$. If I instead assumed a shortage on every down period unconditionally I'd overcount. Second — and this is the one that's easy to get wrong — for *lost* sales the penalty is per unit short, not per unit of *time* short. If I charged $\pi$ per unit-time of being stocked out I'd be penalizing the duration of the outage rather than the volume of demand it swallowed, which is the wrong physical quantity for lost sales. The right charge is $\pi$ times the *number* of unmet demands, which is $D$ times the outage *duration* — and $D(E[T]-Q/D)$ is precisely that. Good, so both the conditioning (only when down) and the per-unit (not per-unit-time) accounting are baked in correctly by going through $E[T]$. Defining the cycle as "between successful shipments" is what makes this clean: the down-time wait is inside the cycle, so its unmet demand is automatically the cycle's shortage.

Put the three together:
$$ E[C] = F + aQ + h\frac{Q^2}{2D} + \pi D\left(E[T]-\frac{Q}{D}\right) = F + aQ + h\frac{Q^2}{2D} - \pi Q + \pi D\,E[T]. $$
Now renewal-reward:
$$ I(Q) = \frac{E[C]}{E[T]} = \pi D + \frac{F + aQ + h\frac{Q^2}{2D} - \pi Q}{E[T]}, $$
where I pulled the $\pi D\,E[T]$ term out of the ratio as a constant $\pi D$. So the order-quantity-dependent part is the numerator $N(Q)=F+aQ+h\frac{Q^2}{2D}-\pi Q$ over $E[T]$. Let me write the whole thing out with $E[T]$ substituted:
$$ I(Q) = \pi D + \frac{F + aQ + \frac{hQ^2}{2D} - \pi Q}{\frac{Q}{D} + A_0\left(1-e^{-(\lambda+\psi)Q/D}\right)}. $$

Let me check the degenerate limit before I trust this at all. Turn off disruptions: $\lambda\to 0$, so $A_0\to 0$ and $E[T]\to Q/D$. Then
$$ I(Q) \to \pi D + \frac{F + aQ + \frac{hQ^2}{2D} - \pi Q}{Q/D} = \pi D + \frac{FD}{Q} + aD + \frac{hQ}{2} - \pi D = \frac{FD}{Q} + aD + \frac{hQ}{2}. $$
The $\pi D$ and the $-\pi D$ cancel — of course they do, there are no stockouts when the supplier never fails — and what's left is exactly Harris. So the disruption model contains the classical model as the no-disruption limit. That's the structural check I needed.

Now, where's the optimum? Let me look at the shape of $I(Q)$. As $Q\to 0$: the $F/Q$-type behavior (numerator $\to F>0$ over $E[T]\to 0$) blows up, so cost is huge for tiny orders — makes sense, you're paying the fixed cost constantly. As $Q\to\infty$: the numerator grows like $\frac{hQ^2}{2D}$, the denominator like $Q/D$, so $I(Q)$ grows linearly in $Q$ — huge holding for giant orders. The limiting behavior gives me a finite positive minimizer, but it does not by itself prove that a search is safe. For that I need the level sets to be intervals: after writing $I(Q)\le \epsilon$ and multiplying by the positive $E[T]$, the inequality becomes one smooth quadratic-plus-exponential expression in $Q$, and the renewal analysis gives the single-trough property I need, namely quasiconvexity rather than ordinary convexity. So a one-dimensional global search such as bisection on the derivative or golden-section search on $I(Q)$ can find $Q^*$ without choosing among local troughs.

But now I hit a wall. Try to actually *solve* $I'(Q)=0$ in closed form. The denominator carries $e^{-(\lambda+\psi)Q/D}$, and when I differentiate the ratio that exponential lands in both the derivative and, through the quotient rule, gets multiplied against polynomial terms in $Q$. There's no way to isolate $Q$ — it's transcendental. So I can get $Q^*$ numerically to any precision I like, and that's fine if all I want is a number. But it's *not* fine if I want to understand how $Q^*$ moves with the parameters, or — the real motivation — if I want to drop this order quantity inside a bigger model (say a whole network of these), where a clean algebraic $Q^*$ would compose and a call to a numerical solver would not. The exact model gives me a value but not a *formula*. I want a formula.

So let me stare at what's actually making it transcendental. It's the single term $e^{-(\lambda+\psi)Q/D}$ sitting in $E[T]$. Everything else is polynomial. Can I get rid of *that one term*? Look at what it does: $E[T]=\frac{Q}{D}+A_0\left(1-e^{-(\lambda+\psi)Q/D}\right)$. The factor $\left(1-e^{-(\lambda+\psi)Q/D}\right)$ is the transient correction in the down-probability, starting from a supplier that is certainly up at the shipment epoch and relaxing toward its stationary down probability. It ramps from $0$ at $Q=0$ up to $1$ as the depletion time grows. Now here's the thing about realistic instances: the recovery rate $\psi$ is typically large relative to the order frequency — disruptions, while they happen, don't last that long compared to the time between orders. If $\psi$ is large, then $(\lambda+\psi)Q/D$ is large even for modest $Q$, and $e^{-(\lambda+\psi)Q/D}\approx 0$. Concretely: if the supplier's mean up-time and the firm's own cycles are on the order of months-to-a-year while down-times are weeks, that exponent is several, and the exponential is only a small tail. So the weighting $\left(1-e^{-(\lambda+\psi)Q/D}\right)\approx 1$ — by the time I reach the reorder instant, the initial certainty that the supplier was up has mostly washed out.

So replace that exponential with $0$. Then
$$ E[T]\approx \frac{Q}{D} + A_0, $$
a *constant* additive buffer $A_0$ tacked onto the classical cycle length. That's the whole trick: the disruption stops being a $Q$-dependent transcendental weighting and becomes a fixed lump of extra expected cycle time. And the approximate cost,
$$ \hat I(Q) = \pi D + \frac{F + aQ + \frac{hQ^2}{2D} - \pi Q}{\frac{Q}{D}+A_0}, $$
is now a ratio of a quadratic over an affine function of $Q$ — and *that* I can minimize in closed form.

Let me do it. I'm minimizing $f(Q)=\frac{N(Q)}{g(Q)}$ with $N=F+aQ+\frac{hQ^2}{2D}-\pi Q$ and $g=\frac{Q}{D}+A_0$. The quotient rule gives $f'=\frac{N'g-Ng'}{g^2}$, so the optimum sets $N'g=Ng'$. Here $g'=\frac{1}{D}$ and $N'=a+\frac{hQ}{D}-\pi$. So
$$ \left(a+\frac{hQ}{D}-\pi\right)\left(\frac{Q}{D}+A_0\right) = \left(F+aQ+\frac{hQ^2}{2D}-\pi Q\right)\frac{1}{D}. $$
Multiply both sides by $D$:
$$ \left(a+\frac{hQ}{D}-\pi\right)\left(Q+A_0 D\right) = F+aQ+\frac{hQ^2}{2D}-\pi Q. $$
Expand the left side:
$$ (a-\pi)Q + (a-\pi)A_0 D + \frac{hQ^2}{D} + hA_0 Q. $$
The right side is $F + (a-\pi)Q + \frac{hQ^2}{2D}$. The $(a-\pi)Q$ terms cancel on both sides, leaving
$$ (a-\pi)A_0 D + \frac{hQ^2}{D} + hA_0 Q = F + \frac{hQ^2}{2D}. $$
Bring everything to one side: $\frac{hQ^2}{D}-\frac{hQ^2}{2D} = \frac{hQ^2}{2D}$, so
$$ \frac{h}{2D}\,Q^2 + hA_0\,Q + \big((a-\pi)A_0 D - F\big) = 0. $$
A quadratic in $Q$. Take the positive root:
$$ \hat Q = \frac{-hA_0 + \sqrt{(hA_0)^2 - 4\cdot\frac{h}{2D}\big((a-\pi)A_0 D - F\big)}}{2\cdot\frac{h}{2D}} = \frac{-hA_0 + \sqrt{h^2A_0^2 + \frac{2h}{D}\big(F+(\pi-a)A_0 D\big)}}{h/D}. $$
In the costly-shortage regime I care about, $\pi>a$, so the constant term $(a-\pi)A_0D-F$ is negative; the discriminant exceeds $h^2A_0^2$, the square root exceeds $hA_0$, and $\hat Q>0$. Then there is exactly one positive root, while the other root is negative and meaningless as an order size. The approximate objective also has the right curvature. Long division gives
$$ \frac{F+(a-\pi)Q+\frac{hQ^2}{2D}}{Q/D+A_0}
=\frac{hQ}{2}+D\left(a-\pi-\frac{hA_0}{2}\right)
+\frac{F+(\pi-a)A_0D+\frac{hA_0^2D}{2}}{Q/D+A_0}. $$
The remainder in the last numerator is positive when $\pi>a$, and $1/(Q/D+A_0)$ has positive second derivative for $Q\ge 0$, so this stationary point is the minimum.

Now the limit check again, because this is where I find out whether the approximation is faithful in spirit. Turn off disruptions: $\lambda\to 0 \Rightarrow A_0\to 0$. The quadratic collapses to $\frac{h}{2D}Q^2 - F = 0$, i.e. $Q^2 = \frac{2FD}{h}$, so $\hat Q\to\sqrt{2FD/h}=Q_E$. The approximation *exactly* recovers the classical EOQ when there are no disruptions. So $\hat Q$ is a genuine generalization: classical EOQ plus a disruption correction that lives entirely in $A_0$.

And the direction of the correction is the right one under the intended economics, but I should state the condition rather than wave at it. Let
$$ P(Q)=\frac{h}{2D}Q^2+hA_0Q+\big((a-\pi)A_0D-F\big) $$
be the quadratic whose positive zero is $\hat Q$. At the classical EOQ, the $\frac{h}{2D}Q_E^2$ term is exactly $F$, so
$$ P(Q_E)=A_0\big(hQ_E+(a-\pi)D\big). $$
Since the leading coefficient is positive, $\hat Q>Q_E$ exactly when this value is negative, i.e.
$$ \pi-a > \frac{hQ_E}{D}. $$
That is the normal disruption-risk regime: the penalty for an unmet unit is more than the avoided purchase cost by enough to justify carrying extra inventory. Then the firm orders more per cycle, hits the bottom less often, and holds $\hat Q-Q_E$ as safety stock against supply unavailability. If shortage is barely costly, the same formula says not to inflate the order blindly; the economics, not the word "disruption," decides the sign.

The last thing I owe myself is a validation plan, not another derivation. I can minimize the exact transcendental $I(Q)$ numerically, compute $\hat Q$ from the quadratic, and compare the order quantities and the resulting costs across frequent-short and rare-long disruption regimes. That check tells me how much accuracy I traded for the closed form. The analytic work already gives the causal structure: a clean $\hat Q$ that reduces to Harris when supply is reliable and, when stockouts are costly enough, inflates smoothly as the supplier becomes less reliable.

Let me write it down as code: the supply primitives ($A_0$, $E[T]$), the exact renewal-reward cost rate, a golden-section minimizer over it (legitimate because the cost is single-troughed), the closed-form $\hat Q$ from the quadratic, and the classical EOQ.

```python
import math

def cycle_buffer_constant(lam, psi):
    # A0 = lambda / (psi (lambda+psi)): the fixed expected wait-weighting that
    # the disruption adds to the cycle. Zero when lambda=0 (no disruptions).
    return lam / (psi * (lam + psi))

def expected_cycle_length(Q, D, lam, psi):
    # E[T] = Q/D + A0 (1 - e^{-(lambda+psi) Q/D}).
    # Q/D is the depletion time; the rest is the expected extra wait when the
    # supplier is down at the reorder instant (phi(Q/D) times mean down-time 1/psi).
    A0 = cycle_buffer_constant(lam, psi)
    return Q / D + A0 * (1.0 - math.exp(-(lam + psi) * Q / D))

def cost_rate(Q, D, F, a, h, pi, lam, psi):
    # Exact long-run average cost per unit time = E[C]/E[T]:
    #   pi*D + (F + a Q + h Q^2/2D - pi Q) / E[T].
    # Holding is the classical triangle h Q^2/2D; the shortage term is folded
    # into pi*D - pi Q/E[T] via the unmet demand D(E[T]-Q/D).
    ET = expected_cycle_length(Q, D, lam, psi)
    return pi * D + (F + a * Q + h * Q * Q / (2.0 * D) - pi * Q) / ET

def optimize_exact(D, F, a, h, pi, lam, psi, lo=1e-9, hi=None, tol=1e-9):
    # Golden-section search; valid because cost_rate is single-troughed in Q.
    if hi is None:
        hi = 50.0 * math.sqrt(2.0 * F * D / h) + D
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
    fc = cost_rate(c, D, F, a, h, pi, lam, psi)
    fd = cost_rate(d, D, F, a, h, pi, lam, psi)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo); fc = cost_rate(c, D, F, a, h, pi, lam, psi)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo); fd = cost_rate(d, D, F, a, h, pi, lam, psi)
    Q = 0.5 * (lo + hi)
    return Q, cost_rate(Q, D, F, a, h, pi, lam, psi)

def approximate_order_quantity(D, F, a, h, pi, lam, psi):
    # Drop e^{-(lambda+psi)Q/D} (small when psi is large): E[T] ~ Q/D + A0.
    # First-order condition becomes the quadratic
    #   (h/2D) Q^2 + (h A0) Q + ((a - pi) A0 D - F) = 0.
    # Positive root; reduces to sqrt(2FD/h) when lambda=0 (A0=0).
    A0 = cycle_buffer_constant(lam, psi)
    qa = h / (2.0 * D); qb = h * A0; qc = (a - pi) * A0 * D - F
    return (-qb + math.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)

def classical_eoq(D, F, h):
    return math.sqrt(2.0 * F * D / h)  # Harris
```

The causal chain, start to finish: an always-available supplier makes the cycle deterministic and gives Harris's $\sqrt{2FD/h}$; let the supplier fail at rate $\lambda$ and recover at rate $\psi$ and the cycle length becomes random, so I move to renewal-reward, cost per cycle over expected cycle length; the cycle is the span between successful shipments, which regenerates because the supplier is up and inventory is $Q$ at each start; the supplier's two-state chain gives $\phi(t)=\frac{\lambda}{\lambda+\psi}(1-e^{-(\lambda+\psi)t})$, and memorylessness makes the expected wait at the bottom $\frac{1}{\psi}\phi(Q/D)$, so $E[T]=Q/D+A_0(1-e^{-(\lambda+\psi)Q/D})$ with $A_0=\frac{\lambda}{\psi(\lambda+\psi)}$; the cost per cycle is fixed-plus-variable ordering, the classical holding triangle $hQ^2/2D$, and a shortage charge $\pi D(E[T]-Q/D)$ that correctly counts unmet *units* only during *actual* down-time; renewal-reward then gives a quasiconvex but transcendental $I(Q)$, minimized numerically; for a closed form I exploit that realistic $\psi$ makes $e^{-(\lambda+\psi)Q/D}\approx 0$, turning $E[T]$ into the constant-buffer $Q/D+A_0$, whose first-order condition is a quadratic whose positive root $\hat Q$ is the order quantity; $\hat Q$ collapses to $Q_E$ when $\lambda=0$, and it becomes a safety-stock expansion above $Q_E$ when the shortage penalty exceeds the avoided unit cost by more than $hQ_E/D$.
