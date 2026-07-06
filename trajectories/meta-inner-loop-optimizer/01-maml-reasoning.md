The inner-loop rule is the whole point, but it bolts onto a two-loop meta-learner, and the simplest
inner rule is the floor I have to start from — so the pain to begin with is just getting a network to
adapt to a five-example task at all. State the problem baldly, and put real numbers on "baldly,"
because the numbers are what make the difficulty concrete. The backbone is CNN4: four Conv→BN→ReLU→
MaxPool blocks at `hidden_size=64` feeding a linear head to five classes. Count it. The first conv is
$3\to64$ with $3\times3$ kernels, $64\cdot3\cdot3\cdot3+64 = 1792$ weights; each of the next three is
$64\to64$, $64\cdot64\cdot9+64 = 36928$; each block carries a BatchNorm pair of $128$; the head maps
the $1600$-dim feature to five logits, $1600\cdot5+5 = 8005$. That sums to about $121\text{K}$
parameters, of which the convolutional body is $\sim113\text{K}$ and the head only $8005$ — the head is
6.6% of the network, the body fourteen times larger. And the data I get to fit this to is a five-way
task with one or five labeled images per class: at 5-shot that is $25$ support points, at 1-shot just
$5$. A hundred-and-twenty-one-thousand-parameter function fit to five points has degrees of freedom to
spare to memorize them exactly and learn nothing that transfers to the fifteen-per-class query points
($75$ points) I am actually scored on. The only thing that can save me is prior experience: I will have
meta-trained across a whole distribution of related five-way episodes first, and the question is what to
extract from that distribution so the next task takes a few gradient steps and does not overfit. The
harness has already committed to the *shape* of the answer — a learned initialization adapted by a few
inner gradient steps — and fixed everything around it. What it leaves me is the inner adaptation rule,
and the floor of that ladder is the dumbest rule that still adapts.

The cheapest thing I know is pretrain-and-fine-tune, and it is worth seeing exactly why it is not the
floor I want. Pool the meta-training episodes, train one network on the pooled loss, and at test time
take a few gradient steps on the new task's support set. But the pooled objective sees the same input
mapped to different labels depending on the (unknown) task, and I can make the damage concrete. Suppose
one image $x$ appears in episode $\mathcal{T}_a$ carrying output-slot label $2$ and in episode
$\mathcal{T}_b$ carrying slot $4$ — which happens constantly, because the five slots are a fresh
relabeling per task. The pooled cross-entropy over both wants a single softmax at $x$ that is
simultaneously good for target $2$ and target $4$; the two gradient contributions push the logit mass
in opposite directions and settle at a compromise that spreads probability across both slots. Averaged
over the whole distribution of relabelings, the loss-minimizing prediction at $x$ drifts toward the
uniform, non-committal output — an initialization deliberately sitting *between* the answers, built to
minimize average loss, which is a different and often actively unhelpful objective than "be a good
launch point for adaptation." Fine-tuning then has to spend its few steps first undoing that studied
indecision before it can commit to *this* task's labels. So fine-tuning from a generic pretrain is not
broken because gradient descent is weak; it is broken because the *initialization* was never chosen for
adaptability. The fix is to choose the initialization *directly* for the property I want.

And the property I want has to be defined against the failure mode I actually face, so name it
precisely. With $5$ support points and a network of $121\text{K}$ parameters — the head alone is $8005$
parameters staring at five examples — the support cross-entropy is radically underdetermined: I can
drive it to essentially zero by memorizing the five points and leave the fifteen-per-class query loss
sitting near the five-way chance level of $0.20$. That gap, near-zero support loss against near-chance
query loss, *is* the overfitting, and no amount of cleverness in the optimizer removes it if the
initialization was chosen to make support loss small. It is removed only if the initialization was
chosen to make the *query* loss small *after* the support step, which is exactly the object I am about
to write down. The budget the harness reserves is the other half of the framing: it caps extra
learnable optimizer state at roughly one scalar per parameter, about $121\text{K}$ numbers. The floor
spends none of it. That untouched headroom is the ruler the rest of the ladder is measured against —
every later rung is a different way to spend it — so it matters to establish first what the network
does with *zero* learned adaptation state.

Before I commit to that, the other two baseline families deserve a look, because I should know whether
the floor is even *allowed* to be something cheaper or cleverer than gradient adaptation. Metric
learners — Prototypical and Matching Networks — sidestep adaptation entirely: embed the support set,
form a class centroid per slot, and classify a query by nearest centroid, so there is no inner gradient
loop at all. They are elegant and often strong at low shot, but they do not fit the substrate I was
handed. The `InnerLoopOptimizer.adapt` contract is "run inner gradient steps on a *clone* and return the
adapted model"; a nearest-centroid rule has nothing to differentiate through and no adapted $\theta$ to
return — it would mean discarding the two-loop machinery the harness froze, not filling in its one open
slot. So metric learning is not a *floor* for this design object; it is a different object. At the far
end sits the learned optimizer — a recurrent network that reads each gradient and emits the update,
meta-learning initialization, direction, and step size jointly. That is the right ambition eventually,
but it is manifestly the *ceiling*, not the floor: it carries a whole recurrent state to train through
the unrolled inner loop, the most expensive and hardest-to-optimize thing in the room. What is left in
between, and what exactly fits the "adapt a clone by inner gradient steps" contract with the *least*
possible learned machinery, is plain gradient descent from a learned start and nothing else. That is
MAML, and it is the floor not by taste but by elimination: it is the unique rule consistent with the
fixed substrate that adds zero learned optimizer state.

Let me write that property down as an objective, because the moment I do, the whole thing either
becomes trainable or falls apart. Take a task $\mathcal{T}_i$. Adaptation is gradient descent on its
support set, one step to start: $\theta_i' = \theta - \alpha\,\nabla_\theta \mathcal{L}^{\text{sup}}
_{\mathcal{T}_i}(f_\theta)$. I do not care how $f_{\theta_i'}$ does on the support points it just
trained on — of course it does well there, that is the overfitting I am fighting. I care how it does
on *held-out* query points from the same task. So evaluate the adapted model's loss on a separate
query set, and *that* is what I want small. Summing over tasks, $\min_\theta \sum_i \mathcal{L}^{
\text{qry}}_{\mathcal{T}_i}(f_{\theta_i'})$ with $\theta_i' = \theta - \alpha\nabla_\theta \mathcal{L}
^{\text{sup}}_{\mathcal{T}_i}(f_\theta)$. Stare at it: the thing optimized is $\theta$, the
initialization, but the loss is evaluated at $\theta_i'$, the *post-adaptation* parameters. The
held-out query loss after fine-tuning is literally the training signal for the initialization. That is
the inversion — instead of "minimize loss and hope fine-tuning works," it is "make fine-tuning work,
and that *is* the loss." The support/query split is load-bearing: evaluate the meta-loss on the same
points I adapted on and I reward $\theta$ for being a place from which I can *memorize* the support,
which is the failure I am trying to avoid. Splitting them means the only way to lower the meta-loss is
an initialization from which a support step produces a model that *generalizes* — adaptation that must
extrapolate, exactly as at meta-test. This is why the fixed loop draws a 15-shot query set every
episode and scores the adapted clone on it; the harness has baked the honest protocol in.

Can I actually optimize this bilevel object? The outer step is just SGD on the meta-loss, $\theta
\leftarrow \theta - \beta\nabla_\theta \sum_i \mathcal{L}^{\text{qry}}_{\mathcal{T}_i}(f_{\theta_i'})$,
so everything reduces to computing the meta-gradient $\nabla_\theta \mathcal{L}^{\text{qry}}(f_{
\theta_i'})$ where $\theta_i'$ itself contains $\theta$ twice — once in the leading term, once inside
the support gradient. I must differentiate through the gradient step. Writing $\theta_i' = \theta -
\alpha\,g(\theta)$ with $g(\theta)=\nabla_\theta\mathcal{L}^{\text{sup}}(f_\theta)$, the chain rule
gives $\nabla_\theta \mathcal{L}^{\text{qry}}(\theta_i') = (\partial\theta_i'/\partial\theta)^\top
\nabla_{\theta'}\mathcal{L}^{\text{qry}}(\theta_i')$, and the Jacobian of the inner step is $\partial
\theta_i'/\partial\theta = I - \alpha\,\nabla_\theta^2 \mathcal{L}^{\text{sup}}(f_\theta)$ — the
identity from the leading $\theta$, minus $\alpha$ times the *support*-loss Hessian, because the
derivative of the gradient is the Hessian. So the meta-gradient is $(I - \alpha\nabla_\theta^2
\mathcal{L}^{\text{sup}})^\top\nabla_{\theta'}\mathcal{L}^{\text{qry}}(\theta_i')$: the query-loss
gradient at the *adapted* point, premultiplied by the Jacobian of the adaptation step.

I do not trust a chain-rule manipulation until I have watched it produce the right number on a case I
can compute by hand, so collapse everything to one scalar parameter and check. Let the support loss be
$\mathcal{L}^{\text{sup}}(\theta)=\tfrac12 a\theta^2$ (curvature $H=a$, minimum at $0$) and the query
loss $\mathcal{L}^{\text{qry}}(\theta)=\tfrac12 b(\theta-c)^2$ (minimum at $c$) — deliberately two
*different* functions, because that is the whole point. The inner step is $\theta' = \theta-\alpha a
\theta = (1-\alpha a)\theta$, so $\partial\theta'/\partial\theta = 1-\alpha a = 1-\alpha H$, matching
the Jacobian claim exactly. The query gradient at the adapted point is $v=\partial\mathcal{L}^{\text{
qry}}/\partial\theta' = b(\theta'-c)$, and the formula predicts $\mathrm{d}\mathcal{L}^{\text{qry}}/
\mathrm{d}\theta = (1-\alpha H)\,v = (1-\alpha a)\,b(\theta'-c)$. Now put numbers on it: $a=2,\;b=1,\;
c=1,\;\theta=1,\;\alpha=0.1$. Then $\theta'=(1-0.2)\cdot1=0.8$, $v=1\cdot(0.8-1)=-0.2$, and the formula
gives $(0.8)(-0.2)=-0.16$. Differentiating the composed scalar directly, $\mathcal{L}^{\text{qry}}(
\theta)=\tfrac12(0.8\theta-1)^2$ so $\mathrm{d}/\mathrm{d}\theta=(0.8\theta-1)(0.8)=(-0.2)(0.8)=-0.16$
at $\theta=1$. They agree, and the agreement is not cosmetic — it confirms the two losses stay
distinct in the formula (the Hessian is of the loss I adapted on, the gradient is of the loss I
evaluate; confuse them and the $-0.16$ would come out wrong). The $-\alpha H$ correction is doing real
work: a step in $\theta$ does not move $\theta_i'$ by the same amount, because moving $\theta$ also
moves the gradient I subtract, and the Hessian is exactly how much the inner gradient bends. Here that
correction is the difference between the naive $b(\theta'-c)=-0.2$ and the true $-0.16$; the factor
$(1-\alpha H)=0.8$ is the outer optimizer accounting for steering the *start* of a gradient step
rather than the end.

I never form the Hessian — I need $H$ times a vector, one extra backward pass, and in practice not even
that: if I build $\theta_i'$ as a node in the graph (subtract $\alpha g$ *keeping the subtraction*),
forward the query set, and call backward to $\theta$, reverse-mode autodiff produces $(I-\alpha H)^\top
\nabla_{\theta'}\mathcal{L}^{\text{qry}}$ for me — provided the inner gradient was taken with
`create_graph=True`. That flag is the entire cost: it keeps the inner gradient itself in the graph so
the outer backward can differentiate through it, roughly doubling the activation memory of the inner
loop, and that is the price of second order. Nothing about the architecture entered any of this; I only
assumed $f_\theta$ is differentiable and trained by gradient descent, which is why the same three lines
work for a conv net under cross-entropy here as for an MLP under MSE. That generality is the point: no
metric, no recurrent optimizer, no extra parameters, just an initialization and ordinary gradient
descent — exactly what makes this the honest floor of the ladder.

If one inner step is good, several unroll: $\theta_i'' = \theta_i' - \alpha\nabla\mathcal{L}^{\text{
sup}}(\theta_i')$, the Jacobian chaining into a product of $(I-\alpha H_k)$ factors, one per step
along the inner trajectory, handled by autodiff through the unrolled loop. This is precisely what the
harness asks for: 5 inner steps at train, 10 at evaluation, each a differentiable `maml_update` over
*all* parameters with the graph retained, and `meta_parameters()` returning `[]` because a fixed
scalar LR carries no learnable state. But the product form is also where I have to be careful, because
a product of five matrices is exactly the kind of thing that explodes. My step-1 edit is therefore the
trivial one — it *is* the scaffold default: compute the support cross-entropy, take its gradient over
every parameter with `create_graph=True`, take one SGD step at `inner_lr` over the whole network,
repeat `n_steps`, expose no meta-parameters. The distilled module is in the answer.

The 5-versus-10 asymmetry between training and evaluation is worth pausing on, because it is a real
constraint I am inheriting rather than a free parameter. During meta-training every inner step is a
factor in the unrolled graph the outer loop must backprop through, so ten steps would double both the
retained-activation memory of `create_graph=True` and the length of the $(I-\alpha H_k)$ product that
can amplify the meta-gradient — five steps is the affordable, stable compromise for computing the
meta-gradient. At evaluation there is no outer backward pass, so I can afford ten steps to squeeze more
adaptation out of the same frozen initialization for free. The catch is that $\theta$ was optimized so
that a *five*-step trajectory generalizes, and eval then runs *ten* — steps six through ten are
extrapolation beyond what the meta-objective ever saw. MAML tolerates this only because the step rule
is stationary: a fixed scalar rate makes step six behave like step one, so the trajectory does not
qualitatively change shape when I extend it. That tolerance is a property of the rigid rule, and it is
worth remembering as a thing the floor gets for free, because any richer, state-carrying step rule
would have to earn it back.

One more piece of the floor is hiding in `model.train()`, and it sharpens the 1-shot fragility. The
BatchNorm layers, in train mode, normalize each forward pass by the *batch* statistics of whatever I
feed them — here the support set. At 5-shot that batch is $25$ images, a usable estimate of per-channel
mean and variance; at 1-shot it is $5$ images, and the normalization statistics are themselves a
high-variance quantity that shifts every inner step as the features move. So at 1-shot the inner loop is
not only stepping a $121\text{K}$-parameter network along a one-sample-per-class gradient, it is doing
so through a normalization whose own scale is wobbling underneath it — a second, compounding source of
the instability that forces the timid rate. This is not something I can fix inside the inner rule
without touching the frozen harness; it is another reason the honest floor at 1-shot has to walk softly.

There is exactly one harness-specific decision that is *not* in the generic story and that I have to
get right, because it is the difference between a floor that runs and one that diverges, and the
Jacobian product is what tells me why. The fixed loop sets a global `inner_lr = 0.5`. Consider a single
curvature direction with support-Hessian eigenvalue $\lambda$: along it, each inner step multiplies the
perturbation by $(1-\alpha\lambda)$, and five steps multiply by $(1-\alpha\lambda)^5$. That factor also
governs how the meta-gradient propagates back through the unrolled loop, so I need $|1-\alpha\lambda|$
to stay near or below $1$ on the directions that matter. Take a moderately sharp direction, $\lambda=6$.
At $\alpha=0.5$: $1-3=-2$, and $|-2|^5 = 32$ — a perturbation along that direction, and the gradient
signal flowing back along it, are amplified thirty-two-fold over five steps. That is not adaptation,
that is divergence, and it is worst precisely at 1-shot, where the support gradient is estimated from a
*single* example per class and is a high-variance direction to begin with — stepping every one of the
$121\text{K}$ parameters by half of a noisy gradient, five times, throws the clone into a region where
the linearization the meta-gradient relies on is meaningless. Now the standard 1-shot remedy, an order
of magnitude smaller: $\alpha=0.01$ gives $1-0.06=0.94$ on that same $\lambda=6$ direction, and
$0.94^5\approx0.73$ — contractive, the trajectory stays in the basin, the five-step product is
well behaved. So vanilla MAML in this harness is not literally "use 0.5 everywhere"; it is shot-aware:
$\alpha = 0.01$ when `N_SHOT == 1`, else $0.5$.

Why is $0.5$ nonetheless fine at 5-shot, rather than merely a survivable evil? Because the instability
above is a two-part failure — a large step size *and* a noisy gradient to take it along — and 5-shot
fixes the second part. Five examples per class instead of one makes the support gradient a five-sample
average per class, a materially steadier estimate of the task direction, so the same $\alpha=0.5$ lands
along a direction that actually points into the task's basin rather than off a noise cliff; it is also
the standard learn2learn benchmark value at this shot count. There is even corroboration in the
harness's own schedule: it runs $60{,}000$ meta-iterations at 1-shot but only $15{,}000$ at 5-shot, a
four-fold cut, which is what I would expect if 5-shot's richer support ($25$ points versus $5$) makes
each meta-update more informative and the whole thing converge faster. So the shot-aware rate is a
property of this benchmark's chosen `inner_lr` interacting with support-set size, not of MAML the
method — the deviation I have to encode, and the only one.

So at step 1 the floor is settled, and I can reason about what it must do — which is the entire point
of running it. MAML adapts *all* parameters with *one fixed scalar* rate and *no learned optimizer
state*. Everything the task distribution might know about *how* to move — which coordinates should
move a lot and which barely, what combined direction generalizes on a fresh task — none of it can be
expressed; the update direction is locked to the raw gradient, the magnitude is one global number, and
the only thing meta-learned is where to start. That is the ceiling here: a carefully chosen launch
point, then a dumb step. I expect this to be a genuinely competitive floor on the 5-shot benchmarks,
where the larger support set makes the rigid step forgiving and the conv features carry the day — so
miniImageNet 5-shot and CIFAR-FS 5-shot should land respectably. I expect 1-shot to be the soft spot:
even at the reduced $\alpha=0.01$, adapting the whole network from a single example per class with a
one-size-fits-all rate is exactly where a learned, per-coordinate step should later pull ahead, so
1-shot accuracy should be the lowest of the three and — because it is walking the knife-edge between a
rate too small to move the discriminative coordinates and one large enough to diverge — the most
exposed, the number most likely to swing seed to seed. Whatever the precise split, the diagnosis is
already pointed at the next rung: MAML has a *rigid-inner-step* problem, not a bad initialization, so
the fix is to keep the initialization-as-objective frame and put some learnable structure into the step
itself — which is what turns the next edit from "one fixed scalar over all parameters" into a
meta-learned per-coordinate step. The feedback table will tell me whether the 1-shot exposure is as
sharp as I think.
