The inner-loop rule is the whole point, but it bolts onto a two-loop meta-learner, and the simplest
inner rule is the floor I have to start from — so the pain to begin with is just getting a network to
adapt to a five-example task at all. State the problem baldly: I have a CNN4 with on the order of a
hundred thousand parameters and a new task with one or five labeled images per class. Fit a
high-capacity net to five points and I have memorized them and learned nothing that transfers to the
held-out query points of the same task. The only thing that can save me is prior experience: I will
have meta-trained across a whole distribution of related five-way episodes first, and the question is
what to extract from that distribution so the next task takes a few gradient steps and does not
overfit. The harness has already committed to the *shape* of the answer — a learned initialization
adapted by a few inner gradient steps — and fixed everything around it. What it leaves me is the inner
adaptation rule, and the floor of that ladder is the dumbest rule that still adapts.

The cheapest thing I know is pretrain-and-fine-tune, and it is worth seeing exactly why it is not the
floor I want. Pool the meta-training episodes, train one network on the pooled loss, and at test time
take a few gradient steps on the new task's support set. But the pooled objective sees the same input
mapped to different labels depending on the (unknown) task, so the loss-minimizing thing is to predict
something like the average target — an initialization built to minimize average loss, which is a
different and often actively unhelpful objective than "be a good launch point for adaptation." So
fine-tuning from a generic pretrain is not broken because gradient descent is weak; it is broken
because the *initialization* was never chosen for adaptability. The fix is to choose the
initialization *directly* for the property I want.

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
gradient at the *adapted* point, premultiplied by the Jacobian of the adaptation step. Two distinct
losses, and I must not confuse them — the Hessian is of the loss I adapted on (support), the gradient
is of the loss I evaluate (query). The $-\alpha H$ correction is doing real work: a step in $\theta$
does not move $\theta_i'$ by the same amount, because moving $\theta$ also moves the gradient I
subtract; the Hessian is exactly how much the inner gradient bends, and $(I-\alpha H)$ propagates that
bending so the outer optimizer accounts for steering the *start* of a gradient step rather than the
end. I never form the Hessian — I need $H$ times a vector, one extra backward pass, and in practice not
even that: if I build $\theta_i'$ as a node in the graph (subtract $\alpha g$ *keeping the
subtraction*), forward the query set, and call backward to $\theta$, reverse-mode autodiff produces
$(I-\alpha H)^\top\nabla_{\theta'}\mathcal{L}^{\text{qry}}$ for me — provided the inner gradient was
taken with `create_graph=True`. Nothing about the architecture entered any of this; I only assumed
$f_\theta$ is differentiable and trained by gradient descent, which is why the same three lines work
for a conv net under cross-entropy here as for an MLP under MSE. That generality is the point: no
metric, no recurrent optimizer, no extra parameters, just an initialization and ordinary gradient
descent — exactly what makes this the honest floor of the ladder.

If one inner step is good, several unroll: $\theta_i'' = \theta_i' - \alpha\nabla\mathcal{L}^{\text{
sup}}(\theta_i')$, the Jacobian chaining into a product of $(I-\alpha H_k)$ factors, one per step
along the inner trajectory, handled by autodiff through the unrolled loop. This is precisely what the
harness asks for: 5 inner steps at train, 10 at evaluation, each a differentiable `maml_update` over
*all* parameters with the graph retained, and `meta_parameters()` returning `[]` because a fixed
scalar LR carries no learnable state. My step-1 edit is therefore the trivial one — it *is* the
scaffold default: `compute the support cross-entropy, take its gradient over every parameter with
create_graph=True, take one SGD step at inner_lr over the whole network, repeat n_steps`, and expose
no meta-parameters. The distilled module is in the answer.

There is exactly one harness-specific decision that is *not* in the generic story and that I have to
get right, because it is the difference between a floor that runs and one that diverges. The fixed
loop sets a global `inner_lr = 0.5`. That is fine when the support set is large enough to buffer
gradient noise — at 5-shot, five examples per class give a steadier support gradient, and 0.5 is the
standard learn2learn benchmark value. But at 1-shot, full-network adaptation with $\alpha=0.5$ is
unstable: a single example per class makes the support gradient a high-variance estimate, and stepping
*every* parameter — all four conv blocks plus BatchNorm plus the head — by half of it blows the clone
off into a region where the meta-gradient through five such steps is meaningless. The standard 1-shot
MAML recipe is an order of magnitude smaller, $\alpha=0.01$, precisely to keep full-network adaptation
in a stable regime. So vanilla MAML in this harness is not literally "use 0.5 everywhere"; it is
shot-aware: $\alpha = 0.01$ when `N_SHOT == 1`, else $0.5$. This is the only deviation from the bare
default, and it is a property of this benchmark's chosen `inner_lr`, not of MAML the method.

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
1-shot accuracy should be the lowest of the three and the most exposed. Whatever the precise split,
the diagnosis is already pointed at the next rung: MAML has a *rigid-inner-step* problem, not a bad
initialization, so the fix is to keep the initialization-as-objective frame and put some learnable
structure into the step itself — which is what turns the next edit from "one fixed scalar over all
parameters" into a meta-learned per-parameter rate.
