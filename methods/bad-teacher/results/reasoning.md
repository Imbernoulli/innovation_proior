Someone has invoked the right to be forgotten. I have a deep classifier trained on the full dataset
`D_c`, and I'm told to remove the influence of a forget set `D_f` — say one whole class — while keeping
everything the model knows about the rest, `D_r`. The clean answer is to retrain from scratch on `D_r`
alone and ship that; that's the gold model, and it's exactly what I want to imitate. But retraining a
deep net costs hours-to-days, and these requests keep coming. I can't retrain per request. So I'm in the
approximate regime: start from the weights I already have and cheaply nudge them so the model *acts like*
the retrained one — lost its specific grip on `D_f`, untouched on `D_r`.

Let me first be honest about what "lost its grip on `D_f`" should mean, because I think this is where it's
easy to go wrong. The crude target is "accuracy on `D_f` goes to zero." But picture what a retrained model
actually does on the forget class. It was never trained on, say, *Boeing aircraft*, but it was trained on
plenty of other aeroplanes; shown a Boeing it will still mostly guess *aeroplane*, with some spread, at
roughly its generalization error. It does *not* confidently shout *banana*. If my unlearned model drives
the Boeing class to confident-wrong — always *animal*, never *aeroplane* — that's not forgetting, that's
teaching the model a new, sharp, *anti*-fact about Boeings. And that sharp anti-fact is itself a fingerprint:
a membership-inference attacker who notices the model is weirdly, confidently wrong on exactly these inputs
has learned that these inputs were specially scrubbed. So zero accuracy isn't the goal and can actively
leak. The goal is *generalization-level uncertainty* on `D_f` — the model should look like it simply never
studied this material, i.e. roughly random / unsure, not adversarially wrong.

Good. Now what tools are on the table, and where do they break. SISA shards the training set and trains a
model per shard so you only retrain the shard that held the deleted point — but that's a decision you have
to make *before* training, with all the checkpoints and the accuracy hit from ensembling many weak shards;
my model is already trained as one monolith, so this is no help to me now. The Fisher-scrubbing line writes
down a weight update — a noise injection sized by the Fisher information, or an NTK linearization of the
training dynamics — that pushes the parameters toward the retrained distribution; but the derivation leans
on the SGD trajectory (so it dictates how I had to have trained) and on Hessian/Fisher approximations
that are heavy and fragile, and the linearized variant trains a whole extra model alongside. UNSIR learns
an error-maximizing noise pattern for the forget class, fine-tunes the model into it to break the class,
then repairs on retain data — fast, but it's class-only (there's no natural noise pattern for a random
scatter of points across classes) and it pushes forget accuracy to *exactly* zero, the confident-wrong
regime I just argued against. Amnesiac logs every batch's update during training and subtracts the ones
that touched `D_f` — but that means storing the entire update history and being tied to training-time
bookkeeping.

Step back and look at what they all have in common as constraints: they each demand something about *how
the model was trained* (SGD-only, sharded, gradient-logged), or they need second-order objects or auxiliary
models, or they only do one mode of forgetting, or they overshoot into confident-wrong. So my wish-list is
the complement of those failures: a method that touches none of the training procedure — takes the trained
model as a black box of weights — needs no extra *trained* model, works for a class *or* a random subset,
and produces the uncertain-not-wrong forget behavior I argued for rather than confident-wrong. Whether any
single mechanism can satisfy all four at once is the open question; let me see what's left when I take them
seriously.

So let me ask the question directly: I have a trained model, and I want a *different* model — one that
behaves a certain prescribed way on `D_f` and another prescribed way on `D_r`. "Make a model behave like a
prescribed target distribution on given inputs" — that's distillation. In distillation a student is trained
to match a teacher's softened output distribution, `softmax(z/T)`, rather than the hard labels; the soft
targets carry the teacher's "dark knowledge," the relative mass on the wrong classes, and the student
copies *whatever distribution the teacher emits.* The thing I keep underusing about distillation is that
the teacher doesn't have to be *good.* It's just a source of target behavior. The student will faithfully
imitate a teacher that is wise, and it will just as faithfully imitate a teacher that is an idiot.

That reframes everything. I want the unlearned model to behave like the *full, competent* model on `D_r`
and like a *model that never learned anything* on `D_f`. So give it two teachers. On retain inputs, distill
from the competent teacher — a frozen copy of my original trained model — and the student keeps every bit of
its retain knowledge. On forget inputs, distill from an *incompetent* teacher: a same-architecture network
with random weights, freshly initialized and never trained.

The whole privacy argument hinges on what that incompetent teacher actually emits, and I should not just
hope. The fear from the UNSIR critique was *confident-wrong*: a teacher that piles most inputs onto one
wrong class at high probability. Does an untrained net do that, or does it sit near uniform? Let me look at
a concrete one. Take `ResNet18`, `pretrained=False`, a 10-way head, push a batch of inputs through it and
read the softmax. What I find: the mean top-1 probability is about `0.157` against a uniform value of
`0.100`, the largest top-1 over the whole batch is `0.184`, and the per-sample entropy averages `2.252`
nats against a maximum of `log 10 = 2.303` — so the output sits at `0.978` of maximum entropy. That is the
diffuse, low-confidence regime, not confident-wrong: no input is anywhere near a `0.9` spike. There is one
honest wrinkle — the *argmax* is not uniform; in my batch `45` of `64` inputs happen to land on the same
class, because the random weights induce a fixed but arbitrary tilt. But that tilt is held at probability
`~0.16`, so it is a faint preference, not a confident assertion, and it is *input-agnostic* rather than a
learned anti-fact about `D_f` specifically. So distilling toward this teacher on `D_f` pushes the student
toward untrained randomness — the generalization-level uncertainty I argued for — without manufacturing the
sharp anti-fact that would leak. I get the privacy-safe forgetting behavior as a consequence of choosing the
teacher to be random rather than malicious, and I now have a measured reason to believe it rather than an
assertion.

Let me name the pieces. Competent / smart teacher `T_s` with parameters θ: a frozen copy of the original
fully-trained model, it has seen all of `D_c`. Incompetent / dumb teacher `T_d` with parameters φ: a frozen,
randomly-initialized copy of the *same architecture*, it has seen nothing. And the student `S`, which is the
model I'm actually producing. Crucial detail for the student's initialization: I do *not* start it random.
I start it at θ — a copy of the original model — because I want to *keep* the retain utility, and the
cheapest way to keep it is to begin from the model that already has it and only nudge the forget region.
Starting random would throw away everything I'm trying to preserve and force me to relearn `D_r` from the
competent teacher from scratch, which is just slow retraining wearing a distillation costume.

Now the objective. Each sample I feed the unlearning procedure carries an unlearning label `l_u` — 1 if
`x ∈ D_f`, 0 if `x ∈ D_r`. The per-sample loss should route the student to the right teacher: when
`l_u = 0` pull `S(x)` toward `T_s(x)`, when `l_u = 1` pull `S(x)` toward `T_d(x)`. Using KL divergence as
the distillation discrepancy, that's

```
L(x, l_u) = (1 - l_u) · KL( T_s(x) ‖ S(x) )  +  l_u · KL( T_d(x) ‖ S(x) ).
```

Because `l_u` is a hard 0/1 selector, exactly one of the two terms is alive for any given sample — on a
retain sample the objective is `KL(T_s ‖ S)`, on a forget sample it's `KL(T_d ‖ S)`. The student learns to
mimic the competent teacher everywhere it's allowed to keep knowledge, and to mimic the random teacher
exactly on the data it must forget. And note what the random teacher does to the *retain-relevant* generic
features: a Boeing shares wings and fuselage with the rest of the aeroplanes, so even as the student
randomizes its specific Boeing response, the broad "aeroplane-ish" structure is held up by the competent
teacher on all the *other* aeroplane samples in `D_r`. The forget signal erases the specific; the retain
signal preserves the generic. That's why I want to feed the procedure all of `D_f` *plus* a slice of `D_r`,
not `D_f` alone — without live retain pressure the random teacher's noise would bleed outward and corrode
nearby retain behavior.

Let me make the loss concrete enough to code, and I have to be careful about how KL is actually evaluated,
because there are sign and direction traps. I work in logit space. Soften with a temperature `T`:

```
t_s = softmax(T_s_logits / T)        # competent teacher distribution
t_d = softmax(T_d_logits / T)        # incompetent teacher distribution
s   = softmax(S_logits / T)          # student distribution
```

The two teacher distributions are fixed targets — the teachers are frozen, no gradients flow into them.
Since `l_u ∈ {0,1}` picks exactly one teacher per sample, I suspect I can fold the selection into a
*single* target distribution per sample,

```
target = l_u · t_d + (1 - l_u) · t_s ,
```

and run one `KL(target ‖ S)`. On paper the algebra is trivial — at `l_u = 0` the mixture is `t_s`, at
`l_u = 1` it is `t_d` — but "the two terms are dead/alive by the selector" is exactly the kind of thing I
get subtly wrong in vectorized code (broadcasting a column label against a `[B, C]` distribution, summing
over the wrong axis), so I want the numbers, not the intuition. I take a 4-sample, 10-class batch of random
teacher and student logits, set labels `[0, 1, 0, 1]`, and compute both the two-term per-sample value
`(1−l)·KL(t_s‖S) + l·KL(t_d‖S)` and the single mixed-target `KL(target‖S)`, summing over classes within
each sample. Two-term gives `[0.2034, 1.0999, 1.3552, 0.0878]`; the single mixed-target gives
`[0.2034, 1.0999, 1.3552, 0.0878]`; the max absolute difference is `0.0`, not merely small. So the collapse
is an identity, not an approximation — it is just the efficient way to vectorize: build `target` once and
call KL once.

Now the direction. I want `KL(target ‖ student) = Σ_i target_i (log target_i − log student_i)`. In the
gradient with respect to the student, the `target_i log target_i` part is a constant, so minimizing this
is minimizing the cross-entropy `−Σ_i target_i log student_i` — pull the student's log-probabilities up
where the target has mass. The framework's KL primitive takes the student's *log*-probabilities as its
first argument and the target *probabilities* as the second, and computes `Σ target · (log target −
input)`; so I pass `log_softmax(S_logits / T)` as the input and `target` as the second argument. This is
not a separate worry from the collapse check above — the values I matched there were computed as exactly
`Σ target · (log target − log_softmax(student))` by hand on one side and `F.kl_div(log_softmax(student),
target)` on the other, and they agreed to `0.0`, which is precisely the evidence that this argument order
realizes `KL(target ‖ student)` and not its reverse. Getting it wrong — passing probabilities where
log-probabilities are expected, or swapping target and student — silently computes a different functional,
so I keep the convention pinned: student goes in as `log_softmax`, teacher mixture goes in as plain
`softmax`.

About the temperature. The usual distillation reason to raise `T` is to *soften* a sharp teacher so its
dark knowledge over wrong classes becomes visible. Here the forget target is the untrained distribution I
just looked at — and it is already nearly uniform, so there may be nothing left to soften. Let me check
rather than guess: on the same untrained `ResNet18` batch, the mean top-1 probability and mean entropy
across temperatures come out as `T=1`: top-1 `0.157`, entropy `2.252`; `T=2`: `0.127`, `2.290`; `T=4`:
`0.113`, `2.299` (max entropy `2.303`). Raising `T` from 1 to 4 buys a move from `0.978` to `0.998` of
maximum entropy — almost nothing, because the teacher is already diffuse. The softening knob is doing real
work only when the teacher is confident, and this one isn't. Meanwhile the *competent* teacher's ordinary
probabilities are exactly the retain behavior I want to copy, sharpness and all, and softening those would
blur the knowledge I'm trying to preserve. So `T = 1` is the natural choice and the setting I keep. If I
were mixing this with a hard-label cross-entropy term I'd have to think about the usual `T^2` scaling that
keeps softened-loss gradients comparable to hard-label gradients — but I'm not. This objective is pure
distillation against the teacher mixture, so there is no `T^2` multiplier in the loss.

One more thing about the reduction over the batch, because it changes the gradient scale. The mathematical
objective is the *per-sample* KL averaged over the batch — sum the KL contribution across classes within
each sample, then average over samples. In PyTorch that exact normalization is `reduction="batchmean"`.
But if I call `F.kl_div` with no reduction argument, the default is `mean`, which averages over every tensor
entry, samples times classes. I want to know the size of the discrepancy before I lean on either, so on the
same 10-class mixed-target batch I evaluate both: `batchmean` gives `0.6866`, default `mean` gives `0.0687`,
and the ratio is `10.000` — exactly the class count `C`. That makes the relationship concrete: `mean =
batchmean / C`. It does not change the KL direction or the selected teacher target, and since the two
differ by a constant factor the *direction* of the gradient is untouched; only its magnitude is scaled by
`1/C`. To mirror the compact implementation I will use the default reduction and remember that the learning
rate is calibrated with that `1/C` scaling. If I switch to `batchmean`, I should treat it as a
normalization change — multiply the effective step by roughly `C` to compensate — and adjust the step size
rather than as a new objective.

Let me also nail down what "randomly initialized" has to mean operationally. The forget-side teacher is not
created inside the loss. It is a separate model with the same output space, left at an ordinary random
initialization and never trained; in the concrete image run that is a same-architecture `ResNet18` with
`pretrained=False`. The competent teacher is the already-trained full model. I put both teachers in `eval()`
mode and forward them under `no_grad`, and the optimizer only owns the student parameters, so the teachers
are fixed targets even though the loss function itself just receives their logits. The student is a separate
model initialized from the original trained weights, because that is the object I will actually update.

The data mechanics can be simpler than paired retain and forget minibatches. Build one unlearning dataset:
first all forget examples with label `1`, then the retained subset with label `0`, and shuffle it in a
loader. Each batch already carries the selector `l_u`. Forward the student on the batch to get logits with
gradients; forward the full trained teacher and the unlearning teacher on the same batch under `no_grad`.
Form the per-sample mixed target from the two teacher softmaxes selected by the label, compute
`F.kl_div(log_softmax(student/T), target)` with PyTorch's default mean reduction, then zero-grad / backward /
step. The optimizer is typically Adam over the student, but the learning rate is a run setting, not a
constant baked into the loss: the sub-class image run uses a smaller rate than the class-level settings.
The invariant is the short mixed forget-plus-retain run. More retained data, more epochs, or a larger step
size all dial the forget set further toward the random teacher, so those knobs control the amount of
randomization rather than changing the objective.

How would I check I landed in the right regime rather than the confident-wrong one — without retraining a
gold model to compare against? The incompetent teacher gives me a free reference for random behavior. I
can measure how close the unlearned model's output distribution on `D_f` is to the incompetent teacher's,
using a symmetric divergence so the comparison is order-free and bounded — Jensen–Shannon,
`JS(p, q) = ½ KL(p‖m) + ½ KL(q‖m)` with `m = (p+q)/2`. Define a forgetting score as
`1 − (1/n_f) Σ_{i=1}^{n_f} JS(M(x_i), T_d(x_i))` over the forget samples: it is high when the model's
forget-set outputs match the untrained teacher and low when the model holds a fixed pattern there. Neither
extreme is automatically the goal — too close to the random teacher can overshoot, too far from it can be
the confident-pattern leak — so the target is wherever a model that never saw `D_f` would sit, which I can
proxy by the score on a held-out test set (a set the model has, by definition, "perfectly unlearned"). This
needs no retrained model, which was the third constraint I set out. It's a diagnostic on top of the update
rule, not part of the update itself.

Let me write the procedure out as code. The loss is the temperature-`T` KL, under PyTorch's `F.kl_div`
convention, from the label-selected teacher mixture to the student's log-softmax; the outer routine builds
the mixed unlearning loader and keeps the two teachers as external fixed models.

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
        return x, y


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample; label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader,
                    optimizer, device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                        epochs=10, optimizer='adam', lr=0.01, batch_size=256, num_workers=32,
                        device='cuda', KL_temperature=1):
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))
```

The chain of it: I can't retrain per request, so I imitate the retrained model cheaply; "forgetting" should
mean generalization-level uncertainty on `D_f`, not confident-wrong, or I leak; distillation copies
*whatever* a teacher emits, so I distill from the original model on retain data and from an *untrained*
random-weight model on forget data; starting the student at the original weights keeps utility while the
hard 0/1 label routes each sample to its teacher, collapsing to one KL against a per-sample teacher
mixture; the random teacher pulls the forget behavior toward the diffuse, near-max-entropy outputs I
measured an untrained net to produce — uncertain, not confidently wrong — and the same random teacher
doubles as a retrained-model-free yardstick (JS distance to it) for checking I landed in the uncertain, not
confidently patterned, regime.
