MAML's numbers told me where the rigid step costs me, and they told me in the split I predicted. On
the two 5-shot benchmarks the floor is genuinely strong: miniImageNet 5-shot at 0.6379 (tight across
seeds, 0.6314–0.6439) and CIFAR-FS 5-shot at 0.7067 (tighter still, 0.7040–0.7104, std 0.0027). The
exposed spot is 1-shot: mean 0.4365, the lowest of the three, and the *least* stable, std 0.0147, seed
123 sagging to 0.4209 while seed 456 reaches 0.4562. I want to read this split as mechanism, not just
rank, so let me do the arithmetic against the five-way chance floor of $0.20$. Above chance, 1-shot
buys $0.2365$, miniImageNet 5-shot $0.4379$, CIFAR-FS 5-shot $0.5067$ — 5-shot extracts about $1.85\times$
the usable signal of 1-shot on the same miniImageNet distribution, so the shot count, not the dataset,
is the axis of difficulty. The instability tells the same story even more sharply when I normalize it:
the *relative* noise, std over above-chance signal, is $0.0147/0.2365 = 6.2\%$ at 1-shot against
$0.0051/0.4379 = 1.2\%$ at miniImageNet 5-shot and $0.0027/0.5067 = 0.5\%$ at CIFAR-FS 5-shot. So 1-shot
is not merely a little noisier — it is roughly five times noisier than miniImageNet 5-shot and twelve
times noisier than CIFAR-FS relative to the signal, and its seed range ($0.0353$) is five and a half
times CIFAR's ($0.0064$). That spread is the tell. I already had to drop $\alpha$ to $0.01$ at 1-shot
just to keep full-network adaptation from diverging on a single example per class; even at that timid
rate, adapting every parameter with one shared scalar is where MAML is both weakest *and* shakiest.
This is not a bad-initialization problem and it is not an outer-loop problem — the meta-loss frame is
right, the launch point is fine on 5-shot. It is a *rigid-inner-step* problem: the only thing the task
distribution gets to shape is where adaptation starts, and nothing about *how* adaptation moves. So
the fix is to keep MAML's initialization-as-objective frame untouched and put learnable structure into
the step itself.

Let me be precise about what an adaptation procedure for a parametric learner even is, because that
decomposition is what tells me which knob to learn. Any gradient-based adaptation is three things:
where it starts (the initialization), which way it moves (the update direction), and how far (the
learning rate). With abundant data the defaults — random start, follow the gradient, small hand-set
rate — are fine. With one example each of those defaults is a liability, and MAML only learned the
first. It froze the second and third: every coordinate steps along the raw gradient, all at the one
scalar $\alpha$. Whatever the task family knows about which coordinates should move a lot and which
barely, and what combined direction actually generalizes on a fresh five-way episode, cannot be
expressed. The 1-shot fragility is the symptom I can now read directly off the relative-noise numbers:
a single shared rate has to be small enough not to blow up the most sensitive coordinate, which means
it is far too small for the coordinates that should move, so adaptation under-fits exactly where it most
needs to move and the result swings with the seed — the $6.2\%$ relative wobble is that knife-edge made
visible. I want to learn the *step*, not just the start.

The slate of existing answers stakes out the corners, and I should cost them out rather than wave at
them. One route learns the *entire* update rule with a recurrent network — the right ambition (learn
init, direction, and rate together), but it pays backprop-through-time over the unrolled inner loop,
storing every intermediate recurrent state. Put the harness's numbers on that: the recurrent state is on
the order of $\dim\theta \approx 121\text{K}$, carried across the $5$ inner steps and differentiated
through, so the retained state scales like $5 \times 121\text{K}$ *on top of* the second-order graph
MAML already keeps, and the recurrence itself is the notoriously hard-to-optimize part — vanishing and
exploding signals over the unroll. The recurrence is what costs the memory; the recurrence is what is
hard to train; the recurrence is the per-step, per-coordinate machine. But the recurrence was never
the goal — the goal was to learn the direction and the rate. So I want the smallest object I can fold
into the gradient step that still buys me a learned direction and a learned, per-coordinate rate, and
that I can train by ordinary backprop with no recurrence at all — because the harness is built for
exactly that: `__init__` lets me create learnable optimizer parameters, `meta_parameters()` hands them
to the same outer Adam that already optimizes $\theta$, and the budget explicitly allows one extra
scalar per model parameter, about $121\text{K}$ numbers, none of which the floor spent.

So sit between the corners and ask what "richer than a scalar rate" means concretely. MAML's step is
$-\alpha\,g$ with $\alpha$ a scalar. The most general linear thing I could put in front of the
gradient is a matrix, $-A\,g$, which can rotate the step arbitrarily off the gradient and rescale
every coordinate. But a full $A$ is $\dim\theta \times \dim\theta$ — for CNN4 that is $121093^2 \approx
1.5\times10^{10}$, fifteen billion numbers, roughly a hundred-and-twenty-thousand times the model
itself, hopeless to store and meta-learn; and if it varied per step it reintroduces exactly the
per-step-per-coordinate cost that sank the recurrent route. The full matrix is out for the same reason
the LSTM was out. That pushes me to the cheapest preconditioner that still does something a scalar
cannot: the *diagonal*. Let $A = \mathrm{diag}(\alpha)$ with $\alpha$ a vector the size of $\theta$. The
step becomes $-\alpha \odot g$, an elementwise product of a learned vector with the gradient, and its
footprint is exactly $\dim\theta = 121\text{K}$ — one scalar per parameter, precisely the reserved
budget, four orders of magnitude below the full matrix.

Let me check this buys both things I want, because if it only buys per-coordinate rates I have not
escaped "follow the gradient." Per-coordinate rate: yes, trivially — entry $\alpha_j$ is coordinate
$j$'s own learning rate, so the sensitive BatchNorm scale and the head weight can move at completely
different magnitudes, which one scalar can never do. And the direction — here is the part I want to be
sure of, so I will make it quantitative. Is $\alpha \odot g$ parallel to $g$? Take the smallest
non-trivial case, a two-coordinate gradient $g=(1,1)$ and rates $\alpha=(0.5,\,0.05)$, a ten-fold
spread of the kind I expect the task family to want (move a discriminative coordinate ten times faster
than a fragile one). Then $\alpha\odot g=(0.5,\,0.05)$, and the cosine to $g$ is
$\frac{0.5+0.05}{\sqrt{0.5^2+0.05^2}\,\sqrt{2}} = \frac{0.55}{0.50249\cdot1.41421} = 0.774$, an angle of
about $39^\circ$. A ten-fold difference between two rates tilts the update thirty-nine degrees off the
raw gradient — that is not a rounding correction, it is a genuinely different direction, and as the
spread grows the tilt approaches the pure fast-coordinate axis. In general $\alpha\odot g$ stays
parallel to $g$ only when the active entries of $\alpha$ are all equal; the instant they differ, the
coordinate rescaling reorients the vector. So the diagonal, which I almost dismissed as "just
per-coordinate rates," already gives me an off-gradient direction for free: the norm of $\alpha \odot
g$ is the effective step size and its orientation is the effective update direction, both rolled into
the one vector $\alpha$. That is exactly the two ingredients MAML's scalar could not reach, with no
matrix and no recurrence.

There is a second reading of the same object that makes me expect it to fix the 1-shot *variance*
specifically, and it comes straight from the stability picture I already worked out for the floor.
Along a coordinate direction with support-curvature $\lambda_j$, an inner step multiplies the
perturbation by $(1-\alpha_j\lambda_j)$, and a *single* shared $\alpha$ has to keep
$|1-\alpha\lambda_{\max}|$ below one for the sharpest coordinate, forcing $\alpha \lesssim 2/\lambda
_{\max}$ — which is far too small for the many coordinates whose $\lambda_j \ll \lambda_{\max}$, so
those barely move at all. That is the under-fit-the-discriminative-directions failure, now stated as
conditioning: one scalar cannot be simultaneously stable on the stiff coordinates and effective on the
soft ones when the curvature spectrum is spread out, which at 1-shot (a single noisy example per class)
it badly is. A per-coordinate $\alpha_j$ tuned toward $\sim 1/\lambda_j$ gives every coordinate its own
well-scaled step, which is exactly diagonal preconditioning of the Hessian. And this predicts the seed
spread should collapse, not just the mean rise: with a single $\alpha$ pinned to the stability
knife-edge, tiny seed-to-seed differences in the meta-trained $\theta$ land coordinates on different
sides of $|1-\alpha\lambda|=1$, which is how I read the $0.0353$ 1-shot range; give each coordinate a
rate that keeps it comfortably contractive and that knife-edge disappears, so the seeds should bunch.
That is the mechanism behind my headline prediction, and it is falsifiable in the std column, not only
the mean.

Is $\alpha$ hand-designed or learned? The hand-designed adaptive optimizers — AdaGrad, RMSProp, Adam
— also put a per-coordinate rescaling on the gradient, set from the *gradient history* by a fixed
formula. That tells me the *shape* (per-coordinate gradient rescaling) is right; people have found it
works. But their machinery needs a history to accumulate, and I do not have one — five inner steps,
one example per class, no long stream of gradients to take a running norm over. A running second-moment
estimate from five noisy 1-shot gradients would be as unreliable as the gradients themselves. So I
cannot get $\alpha$ from history. What I *do* have is the task distribution. So instead of computing
$\alpha$ from a gradient history, I learn it across tasks: let $\alpha$ be a free vector of
meta-parameters that meta-training tunes so that, on a new task, $n$ steps of $-\alpha \odot g$ land
somewhere that generalizes. $\alpha$ becomes a learned encoding of "for this family of five-way
episodes, how far and which combined way should each coordinate move when the support gradient points
this way" — the same ambition as the recurrent route's learned step, distilled into a single static
vector with no recurrence to pay for.

And $\alpha$ is differentiable, which is the whole reason for choosing this shape. So I fold it
straight into the same query-loss-after-adaptation objective MAML used, except now I meta-learn
$\theta$ *and* $\alpha$ together: $\theta_i' = \theta - \alpha \odot \nabla\mathcal{L}^{\text{sup}}_i
(\theta)$, and $\min_{\theta,\alpha}\, \mathbb{E}_{\mathcal{T}}[\mathcal{L}^{\text{qry}}(\theta -
\alpha \odot \nabla\mathcal{L}^{\text{sup}}(\theta))]$. Before I trust that the *same* outer Adam can
carry $\alpha$, I want to see its gradient come out right on a case I can compute by hand, and I want
to see it is genuinely cheaper than the $\theta$ path. For one task, with $g=\nabla_\theta\mathcal{L}
^{\text{sup}}$, $H=\nabla_\theta^2\mathcal{L}^{\text{sup}}$, $\theta'=\theta-\alpha\odot g$, and
$v=\nabla_{\theta'}\mathcal{L}^{\text{qry}}$, the chain rule gives $\partial\mathcal{L}^{\text{qry}}/
\partial\alpha = -v\odot g$ and $\partial\mathcal{L}^{\text{qry}}/\partial\theta = (I - \mathrm{diag}
(\alpha)H)^\top v$. The $\alpha$-line has *no Hessian* — $\alpha$ enters $\theta'$ linearly through the
elementwise product, so its gradient is a plain elementwise product $-v\odot g$, cheaper than the
$\theta$-line's Hessian-vector path, which is reassuring for the "no extra second order" claim. Now the
hand check: reuse the scalar toy, $\mathcal{L}^{\text{sup}}=\tfrac12 a\theta^2$ so $g=a\theta$, and
$\mathcal{L}^{\text{qry}}=\tfrac12 b(\theta-c)^2$, with $a=2,\,b=1,\,c=1,\,\theta=1,\,\alpha=0.1$. Then
$g=2$, $\theta'=1-\alpha a\theta=1-0.2=0.8$, $v=b(\theta'-c)=-0.2$, and the formula predicts
$\partial\mathcal{L}^{\text{qry}}/\partial\alpha = -v\,g = -(-0.2)(2) = 0.4$. Directly, $\theta'(\alpha)=
1-2\alpha$, so $\mathcal{L}^{\text{qry}}=\tfrac12(1-2\alpha-1)^2 = 2\alpha^2$ and
$\mathrm{d}/\mathrm{d}\alpha = 4\alpha = 0.4$ at $\alpha=0.1$. They match, and the sign is mechanistically
right: the support minimum is at $0$, the query minimum at $1$, so stepping toward $0$ overshoots away
from $1$, the gradient on $\alpha$ is positive, and the outer loop is correctly told to *shrink* this
rate — which is precisely the per-coordinate discrimination a single scalar cannot make. So the *same*
outer Adam optimizes both, by backprop through one (or five) inner steps with the graph kept — no BPTT
through an optimizer network, no stored recurrent states, just one extra tensor per parameter. The
thing that made the recurrent route unscalable is simply gone, because I replaced "a network that emits
the step" with "a learned vector that scales the gradient."

Now ground this in *this task's* edit surface, because the harness dictates the literal form and it is
not the generic one. The contract is `__init__`, `adapt`, `meta_parameters`. In `__init__` I create
the learnable rates as an `nn.ParameterList` — one `nn.Parameter` per model parameter, each
`torch.ones_like(p) * inner_lr`, so every entry starts at the default 0.5 and the rates are uniform at
initialization. That uniform start is deliberate and it is the right one: at the start of meta-training
I have no reason to favor any coordinate, and I want to begin from behavior I trust — which is MAML,
uniform small steps — and let meta-training pull the entries apart as it discovers which coordinates
want bigger or differently-signed steps. Setting every entry of $\alpha$ to the same constant *is*
MAML; the method differentiates from there. For the inner step, the harness gives me `l2l.update_module
(model, updates=[...])`, a differentiable per-parameter $p \leftarrow p + u$ swap that takes one update
per parameter in order. So I compute the support cross-entropy, take `torch.autograd.grad` over all
parameters with the graph retained, build `updates = [-lr * g for g, lr in zip(grads, self.lrs)]` —
the elementwise $-\alpha \odot g$, one entry per parameter — and call `update_module`. That is the one
load-bearing difference from MAML's edit: MAML used `maml_update(model, lr=scalar, grads=grads)` with a
single scalar; Meta-SGD uses `update_module` with a *list* of per-parameter products, because the rate
is now a tensor, not a number. Finally `meta_parameters()` returns `list(self.lrs.parameters())` so the
outer Adam optimizes $\alpha$ alongside $\theta$ — the harness sums them into `all_meta_params` and the
budget check (one scalar per model parameter, $\sim121\text{K}$) is exactly Meta-SGD's footprint. The full
scaffold module is in the answer.

Two design choices inside that form are not forced by the contract and I should make them deliberately.
First, is $\alpha$ shared across the inner steps or fresh per step? A per-step $\alpha_t$ would let the
first step move differently from the fifth, which is genuinely more expressive — but it costs $5\times
121\text{K} \approx 605\text{K}$ meta-parameters, five times over the reserved budget, and, worse, a
schedule of per-step rates *is* a small step-indexed recurrent controller, exactly the machine I threw
out for being hard to train and impossible to reuse at evaluation (where the loop runs ten steps, not
five, so per-step rates trained for steps 1–5 have nothing to say about steps 6–10). A single static
$\alpha$ reused at every step sidesteps both problems and extends cleanly to the ten-step eval loop, so
I keep it static. Second, do I constrain $\alpha$ to be positive? Nothing in the elementwise product
requires it, and I deliberately leave the `nn.Parameter` unconstrained. A negative entry $\alpha_j<0$
steps that coordinate *up* its own gradient, which sounds pathological until I remember the direction
argument: a diagonal with a sign flip tilts the update past ninety degrees off the raw gradient, so the
freedom to go negative is precisely what lets the learned direction depart from "descend the support
loss" on the coordinates where descending it fastest is what overfits. Meta-training will decide; my job
is only to not forbid it. Both choices — static across steps, unconstrained sign — fall out of the same
two principles that chose the diagonal in the first place: stay inside the budget, and buy the learned
direction without buying a recurrence.

Let me confirm the relations so I know I have generalized MAML and not built a third unrelated gadget.
Freeze $\alpha$ to a single constant in every coordinate and stop learning it: $\alpha \odot g = c\,g$,
$\theta'=\theta-c\,g$ — exactly MAML, plain SGD at a scalar rate, meta-learning only $\theta$. So MAML
is the point in this space where $\alpha$ is uniform and fixed; I have strictly generalized it by
letting $\alpha$ be non-uniform and learned. The recurrent route is the expensive way to chase the
same three learned ingredients, which I have replaced with the static diagonal. And the diagonal is the
right rank, not a cop-out — the two cheaper alternatives fail for reasons I can name in numbers. A
per-layer scalar would attach one rate to each parameter tensor; CNN4 has on the order of a dozen or
two such tensors (four conv weights, their biases, the BatchNorm pairs, the head weight and bias), so a
per-layer rule spends a couple of dozen numbers and cannot move two weights *within* a single conv
block at different rates — but within-layer is exactly where the per-coordinate structure lives, so it
throws away the whole point to save a hundred-thousand numbers I am allowed to spend. The full matrix
is the other extreme, the fifteen-billion-number object, most expressive but $\dim\theta^2$ and
per-step-costly. The diagonal is the unique middle that is linear in memory, per-coordinate, and
off-gradient.

So the delta from step 1 is concrete: where MAML applied one shared scalar to every parameter via
`maml_update`, I attach a learnable per-parameter rate vector $\alpha$ (a `ParameterList` of
`ones_like(p)*0.5`), step each parameter by $-\alpha \odot g$ via `update_module`, and hand $\alpha$ to
the outer loop as meta-parameters. Reading MAML's shape, here is what I expect and where I am unsure.
The biggest win should be at **1-shot**, because that is precisely where MAML's single shared rate was
both lowest (0.4365) and shakiest (the $6.2\%$ relative wobble, std 0.0147): a learned per-coordinate
rate can keep the sensitive coordinates timid while letting the discriminative ones move, so 1-shot
mean should rise above 0.4365 and, if the diagnosis is right, the seed spread should tighten — the
learned $\alpha$ removes the knife-edge that the single scalar had to walk, so I would expect the std
to fall well under half of MAML's. Where I am genuinely unsure is the **5-shot** benchmarks: MAML was
already strong there (0.6379 mini, 0.7067 cifar) because the larger support set made the rigid step
forgiving, so the extra capacity of a per-parameter rate has less to fix and could even hurt if the
larger meta-parameter set overfits the meta-training distribution — a non-uniform $\alpha$ tuned to
training episodes need not transfer as cleanly as one shared scalar, and the extra $121\text{K}$
meta-parameters are $121\text{K}$ new ways to fit the wrong thing. The falsifiable claim is therefore
directional and uneven: 1-shot should improve clearly (mean up, std down) while 5-shot may stay flat or
slip, and if it slips the lesson is that the ceiling at 5-shot was never the step's expressiveness — in
which case the next rung is not "more capacity in the step" but a different axis entirely, *which*
parameters to adapt at all. The 5-shot columns of the next table are the ones I will read first.
