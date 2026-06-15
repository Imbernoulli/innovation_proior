MAML's numbers told me where the rigid step costs me, and they told me in the split I predicted. On
the two 5-shot benchmarks the floor is genuinely strong: miniImageNet 5-shot at 0.6379 (tight across
seeds, 0.6314–0.6439) and CIFAR-FS 5-shot at 0.7067 (tighter still, 0.7040–0.7104, std 0.0027). With
five examples per class buffering the support gradient, a single global rate over the whole network is
forgiving and the conv features carry it — exactly as I expected. The exposed spot is 1-shot: mean
0.4365, the lowest of the three, and the *least* stable, std 0.0147, seed 123 sagging to 0.4209 while
seed 456 reaches 0.4562. That spread is the tell. I already had to drop $\alpha$ to 0.01 at 1-shot
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
expressed. The 1-shot fragility is the symptom — a single shared rate has to be small enough not to
blow up the most sensitive coordinate, which means it is far too small for the coordinates that should
move, so adaptation under-fits exactly where it most needs to move and the result swings with the
seed. I want to learn the *step*, not just the start.

The slate of existing answers stakes out the corners. One route learns the *entire* update rule with a
recurrent network — the right ambition (learn init, direction, and rate together), but it pays
backprop-through-time over the unrolled inner loop, storing every intermediate recurrent state, a cost
on the order of (steps × state-size × $\dim\theta$) that simply does not fit a conv learner and is
hard to optimize on top of that. The recurrence is what costs the memory; the recurrence is what is
hard to train; the recurrence is the per-step, per-coordinate machine. But the recurrence was never
the goal — the goal was to learn the direction and the rate. So I want the smallest object I can fold
into the gradient step that still buys me a learned direction and a learned, per-coordinate rate, and
that I can train by ordinary backprop with no recurrence at all — because the harness is built for
exactly that: `__init__` lets me create learnable optimizer parameters, `meta_parameters()` hands them
to the same outer Adam that already optimizes $\theta$, and the budget explicitly allows one extra
scalar per model parameter.

So sit between the corners and ask what "richer than a scalar rate" means concretely. MAML's step is
$-\alpha\,g$ with $\alpha$ a scalar. The most general linear thing I could put in front of the
gradient is a matrix, $-A\,g$, which can rotate the step arbitrarily off the gradient and rescale
every coordinate. But a full $A$ is $\dim\theta \times \dim\theta$ — for CNN4 that is the square of a
hundred-thousand-dimensional vector, hopeless to store and meta-learn, and if it varied per step it
reintroduces exactly the per-step-per-coordinate cost that sank the recurrent route. The full matrix
is out for the same reason the LSTM was out. That pushes me to the cheapest preconditioner that still
does something a scalar cannot: the *diagonal*. Let $A = \mathrm{diag}(\alpha)$ with $\alpha$ a vector
the size of $\theta$. The step becomes $-\alpha \odot g$, an elementwise product of a learned vector
with the gradient.

Let me check this buys both things I want, because if it only buys per-coordinate rates I have not
escaped "follow the gradient." Per-coordinate rate: yes, trivially — entry $\alpha_j$ is coordinate
$j$'s own learning rate, so the sensitive BatchNorm scale and the head weight can move at completely
different magnitudes, which one scalar can never do. And the direction — here is the part I want to be
sure of. Is $\alpha \odot g$ parallel to $g$? For a nonzero gradient with at least two active
coordinates, it stays parallel only when the active entries of $\alpha$ are all the same scalar. Once
those entries differ, the coordinate rescaling *tilts* the vector — $\alpha \odot g$ points in a
genuinely different direction than $g$. So the diagonal, which I almost dismissed as "just
per-coordinate rates," already gives me an off-gradient direction for free: the norm of $\alpha \odot
g$ is the effective step size and its orientation is the effective update direction, both rolled into
the one vector $\alpha$. That is exactly the two ingredients MAML's scalar could not reach, with no
matrix and no recurrence — $\alpha$ is just a tensor the same shape as $\theta$, one learnable scalar
per parameter, precisely the budget the harness reserved.

Is $\alpha$ hand-designed or learned? The hand-designed adaptive optimizers — AdaGrad, RMSProp, Adam
— also put a per-coordinate rescaling on the gradient, set from the *gradient history* by a fixed
formula. That tells me the *shape* (per-coordinate gradient rescaling) is right; people have found it
works. But their machinery needs a history to accumulate, and I do not have one — five inner steps,
one example per class, no long stream of gradients to take a running norm over. So I cannot get
$\alpha$ from history. What I *do* have is the task distribution. So instead of computing $\alpha$ from
a gradient history, I learn it across tasks: let $\alpha$ be a free vector of meta-parameters that
meta-training tunes so that, on a new task, $n$ steps of $-\alpha \odot g$ land somewhere that
generalizes. $\alpha$ becomes a learned encoding of "for this family of five-way episodes, how far and
which combined way should each coordinate move when the support gradient points this way" — the same
ambition as the recurrent route's learned step, distilled into a single static vector with no
recurrence to pay for.

And $\alpha$ is differentiable, which is the whole reason for choosing this shape. So I fold it
straight into the same query-loss-after-adaptation objective MAML used, except now I meta-learn
$\theta$ *and* $\alpha$ together: $\theta_i' = \theta - \alpha \odot \nabla\mathcal{L}^{\text{sup}}_i
(\theta)$, and $\min_{\theta,\alpha}\, \mathbb{E}_{\mathcal{T}}[\mathcal{L}^{\text{qry}}(\theta -
\alpha \odot \nabla\mathcal{L}^{\text{sup}}(\theta))]$. For one task, with $g=\nabla_\theta\mathcal{L}
^{\text{sup}}$, $H=\nabla_\theta^2\mathcal{L}^{\text{sup}}$, $\theta'=\theta-\alpha\odot g$, and
$v=\nabla_{\theta'}\mathcal{L}^{\text{qry}}$, the chain rule gives $\partial\mathcal{L}^{\text{qry}}/
\partial\alpha = -v\odot g$ and $\partial\mathcal{L}^{\text{qry}}/\partial\theta = (I - \mathrm{diag}
(\alpha)H)^\top v$. The $\theta$-line is MAML's same through-the-inner-gradient path; the $\alpha$-line
is simpler because $\alpha$ enters $\theta'$ linearly through the elementwise product. So the *same*
outer Adam optimizes both, by backprop through one (or five) inner steps with the graph kept — no
BPTT through an optimizer network, no stored recurrent states, just one extra tensor per parameter.
The thing that made the recurrent route unscalable is simply gone, because I replaced "a network that
emits the step" with "a learned vector that scales the gradient."

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
budget check (one scalar per model parameter) is exactly Meta-SGD's footprint. The full scaffold module
is in the answer.

Let me confirm the relations so I know I have generalized MAML and not built a third unrelated gadget.
Freeze $\alpha$ to a single constant in every coordinate and stop learning it: $\alpha \odot g = c\,g$,
$\theta'=\theta-c\,g$ — exactly MAML, plain SGD at a scalar rate, meta-learning only $\theta$. So MAML
is the point in this space where $\alpha$ is uniform and fixed; I have strictly generalized it by
letting $\alpha$ be non-uniform and learned. The recurrent route is the expensive way to chase the
same three learned ingredients, which I have replaced with the static diagonal. And the diagonal is the
right rank, not a cop-out: a per-layer scalar is cheaper but can only rescale a whole layer uniformly,
it cannot move two weights *within* a conv block at different rates, and within-layer is exactly where
the per-coordinate structure lives; the full matrix is the other extreme, most expressive but $\dim
\theta^2$ and per-step-costly. The diagonal is the unique middle that is linear in memory,
per-coordinate, and off-gradient.

So the delta from step 1 is concrete: where MAML applied one shared scalar to every parameter via
`maml_update`, I attach a learnable per-parameter rate vector $\alpha$ (a `ParameterList` of
`ones_like(p)*0.5`), step each parameter by $-\alpha \odot g$ via `update_module`, and hand $\alpha$ to
the outer loop as meta-parameters. Reading MAML's shape, here is what I expect and where I am unsure.
The biggest win should be at **1-shot**, because that is precisely where MAML's single shared rate was
both lowest (0.4365) and shakiest (std 0.0147): a learned per-coordinate rate can keep the sensitive
coordinates timid while letting the discriminative ones move, so 1-shot mean should rise above 0.4365
and, if the diagnosis is right, the seed spread should tighten — the learned $\alpha$ removes the
knife-edge that the single scalar had to walk. Where I am genuinely unsure is the **5-shot**
benchmarks: MAML was already strong there (0.6379 mini, 0.7067 cifar) because the larger support set
made the rigid step forgiving, so the extra capacity of a per-parameter rate has less to fix and could
even hurt if the larger meta-parameter set overfits the meta-training distribution — a non-uniform
$\alpha$ tuned to training episodes need not transfer as cleanly as one shared scalar. The falsifiable
claim is therefore directional and uneven: 1-shot should improve clearly (mean up, std down) while
5-shot may stay flat or slip, in which case the next rung is not "more capacity in the step" but a
different axis — *which* parameters to adapt at all.
