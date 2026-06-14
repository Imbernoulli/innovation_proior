## Research question

We have a large, well-trained autoregressive language-model teacher and a much smaller student
we actually want to deploy. The student must reproduce the teacher's behavior at a fraction of the
parameters, memory, and decoding cost. Knowledge distillation gives us the mechanism — train the
student to match the teacher's per-token predictive distribution rather than only the one-hot
next token — but the quality of the distilled student is governed entirely by the *objective*: the
divergence we minimize between the teacher distribution `p` and the student distribution `q_theta`,
position by position over the sequence.

The hard part is that the teacher and the student are very *different* models, and that difference
shows up in three coupled ways that the choice of objective has to contend with. First, a **capacity
gap**: the teacher has far more parameters and can represent distributions the small student
provably cannot. Second, **mode averaging**: a student told to cover every region of probability
mass the teacher places will, when it lacks the capacity to do so faithfully, smear its mass thinly
across all of them and produce an oversmoothed, low-accuracy distribution. Third, **mode collapse**:
a student told instead to stay only where the teacher is confident will pile all of its mass on a
few dominant modes and drop the rest of the teacher's structure. The precise goal is a distillation
objective for autoregressive LMs that transfers knowledge from a teacher that may be *much* larger
than the student — without the student either oversmoothing to cover an unreachable teacher or
collapsing onto a sliver of it — and that does so cheaply, ideally from teacher logits alone without
an expensive sampling loop. Each existing objective below sits at one end of this averaging/collapse
axis, or it relies on a fixed comparison distribution whose signal can be poorly matched to the
student under a large capacity gap. Closing that gap is the problem.

## Background

By this point, distilling large LMs into compact ones is a mainstream route to deployable models,
and the dominant recipe descends from Hinton et al. (2015): match the teacher's softmax
distribution. A language model is a probability distribution over token sequences
`y = (y_1, ..., y_S)`, factorized autoregressively `p(y) = prod_s p(y_s | y^{<s})`, with each
conditional a softmax over the vocabulary of the model's logits. Distillation replaces (or augments)
the one-hot training target with the teacher's full conditional `p(y_s | y^{<s})` and minimizes a
per-position divergence between teacher and student, averaged over the `S` positions of the
sequence.

The load-bearing concepts are about *which* divergence and *what target*:

- **The forward/reverse KL asymmetry.** Forward KL, `KL(p || q) = sum_y p log(p/q)`, is
  **mass-covering** (a.k.a. mean-seeking): it is finite only where `q` puts mass wherever `p` does,
  so minimizing it forces the student to spread mass over every teacher mode. When the student lacks
  the capacity to represent that full distribution, the result is oversmoothing — the
  **mode-averaging** failure. Reverse KL, `KL(q || p)`, is **mode-seeking**: it heavily penalizes
  the student for putting mass where the teacher has little, so the student concentrates on the
  teacher's dominant modes — and a low-capacity student driven this way piles onto a few of them and
  drops the rest, the **mode-collapse** failure. These are the same two pathologies named above,
  now tied to the geometry of the two KL directions.

- **The curse of the capacity gap (a diagnostic phenomenon).** Intuition says a stronger teacher
  should produce a stronger student. In practice, beyond a point, *enlarging the teacher degrades
  the student*: distilling from a 410M, 1B, 2.8B, 6.9B family into a fixed small student, accuracy
  can fall as the teacher grows, and fixed-target objectives that look fine with a comparable teacher
  break down when the teacher is much larger. Mirzadeh et al. (2019), Cho & Hariharan (2019), and
  Zhang et al. (2023) report this across settings: a target distribution can be too far from the
  student's reachable set regardless of how good that target is in the abstract.

- **Self-distillation as a regularizer, and its theoretical ceiling.** Distilling a model into a
  copy of *itself* — feeding the model's own predictions back as labels and refitting — is known to
  improve generalization. Mobahi, Farajtabar & Bartlett (2020) make this precise for least-square
  regression in the interpolation regime: minimize a kernel regularizer `R(f) = integral u(x,x') f(x) f(x')`
  subject to fitting the training labels to tolerance `eps`. The solution is a nonlinear ridge
  estimator; stacking predictions over the training inputs gives `f = V^T D (lambda I + D)^{-1} V y`,
  where the kernel matrix `G = V^T D V` has orthogonal `V` and positive diagonal `D = diag(d_i)`,
  and the Lagrange multiplier obeys `lambda = alpha sqrt(N eps) / (||y|| - sqrt(N eps))` for some
  `alpha in [d_min, d_max]`. Each round of self-distillation applies the diagonal contraction
  `A = D (lambda I + D)^{-1}` (singular values in `(0,1)`) to the label vector, so in the rotated
  coordinates `z = V y` the signal shrinks every round: `z_{t} = D (c_{t-1} I + D)^{-1} z_{t-1}`.
  This contraction *sparsifies* the representation — it amplifies the top-eigenvalue direction and
  suppresses the rest, which is the regularization benefit — but it is also a one-way shrinkage. The
  non-collapse condition is `||z_t|| > sqrt(N eps)`; once the signal falls below that floor the
  estimator collapses to the zero function and stays there. The number of guaranteed safe rounds is
  about `(r_0 - 1)/kappa` with `r_0 = ||y_0|| / sqrt(N eps)` and condition number
  `kappa = d_max / d_min`, and for enough rounds self-distillation *inevitably* collapses (their
  Proposition 4). A recursion whose labels are built only from the model's own shrinking predictions
  cannot maintain its signal indefinitely.

## Baselines

**Forward-KL distillation (Hinton, Vinyals & Dean, 2015).** The original KD objective, written for
LMs as `J_KL(p, q_theta) = (1/S) sum_s sum_{y_s} p(y_s|y^{<s}) log( p(y_s|y^{<s}) / q_theta(y_s|y^{<s}) )`.
The student is trained to match the teacher's full soft distribution at every position. Core idea:
the teacher's soft probabilities carry "dark knowledge" — relative weights over wrong answers — that
a one-hot label lacks. **Limitation:** forward KL is mass-covering, so under a capacity gap the
student oversmooths to cover all teacher modes and loses accuracy (mode averaging).

**Reverse-KL and f-divergence distillation (Wen et al., 2023).** Replace forward KL with reverse KL
`J_RKL(p, q_theta) = J_KL(q_theta, p)`, and more generally consider f-divergences such as total
variation distance `J_TVD(p, q) = (1/2) sum_y |p(y) - q(y)|`. Core idea: a mode-seeking objective
keeps the student honest where the teacher is confident, avoiding the oversmoothing of forward KL.
**Limitation:** reverse KL is mode-seeking, so a low-capacity student concentrates on a few dominant
teacher modes and drops the rest (mode collapse). The two KL directions trade one pathology for the
other; neither sits in between.

**Generalized JSD distillation (Agarwal et al., ICLR 2024).** Interpolate between the two KL
directions with a mixture distribution in probability space. With `0 < beta < 1` and
`r = beta p + (1 - beta) q_theta`,
`J_GJSD(p, q_theta) = beta KL(p || r) + (1 - beta) KL(q_theta || r)`,
which behaves like forward KL as `beta -> 0` and reverse KL as `beta -> 1`. This sits the objective
*between* averaging and collapse, tuned by a single coefficient. The same line pairs this with an
on-policy / student-generated-output regime — cast distillation as imitation learning with an
interactive expert, sample sequences from the current student, and have the teacher label them — to
fix the train/inference mismatch of autoregressive models. **Limitations:** the mixture coefficient
`beta` is *fixed* throughout training and the probability-space mixture must be chosen upfront. If
the teacher is far outside the small student's representable family, the comparison distribution can
remain too teacher-dominated for the student to use well. The on-policy variant adapts the data
distribution via student samples, but pays a heavy cost: generating student rollouts every step is
expensive at LLM scale.

**Skew-KL distillation (Ko et al., ICML 2024).** Mix teacher and student in probability space with a
fixed skew `alpha`, and have the *teacher* teach the mixture:
`J_SKL(p, q_theta) = KL(p, alpha p + (1 - alpha) q_theta)`, with a reverse variant
`J_SRKL(p, q_theta) = KL(q_theta, (1 - alpha) p + alpha q_theta)`, typically `alpha ~ 0.1`. Core
idea: blending the student into the comparison distribution keeps the KL ratio's denominator away
from zero, which bounds the gradient norm and stabilizes optimization (the bare KL gradient explodes
as the student probability of a token goes to zero). It is also paired with an adaptive off-policy
scheduler to use student-generated outputs sparingly. **Limitations:** the mixing coefficient is
again *fixed* over training, and the knowledge flow is indirect — the teacher is taught to match the
mixture rather than the student being taught directly — while the interpolation ratio is constant, so
the comparison geometry is set before training begins.

The recurring gap across all four: the main comparison object is a fixed teacher distribution, a
fixed teacher/student blend, or an expensive sampled interaction. Under a large capacity gap, the
cheap fixed-objective choices can expose the small student to a signal that is either too broad
(averaging), too narrow (collapse), or too difficult to fit.

## Evaluation settings

The yardsticks already in use for LM distillation:

- **Instruction tuning** — distill an instruction-following student on a chat/instruction corpus
  (e.g. UltraChat-200k), selecting checkpoints by ROUGE-L on a held-out split, then evaluating
  instruction-following quality. Teacher/student pairs spanning families and sizes.
- **Pre-training distillation** — distill on a large web/text corpus (e.g. a slice of SmolLM-Corpus,
  on the order of tens of billions of tokens), one epoch, AdamW with a cosine schedule.
- **Capacity-gap sweep** — hold the small student fixed and vary the teacher across a size family
  (e.g. 410M / 1B / 2.8B / 6.9B) to measure how an objective's distilled quality changes as the gap
  widens. This is the protocol that exposes the curse of the capacity gap.
- **Math-reasoning distillation** — a small base student and a math-tuned teacher, training prompts
  drawn from a math corpus; evaluate on reasoning benchmarks (grade-school word problems,
  competition-style problems, contest sets), extracting the final boxed answer and grading by a
  symbolic checker. Greedy decoding for the large splits; averaging several samples per problem for
  the small contest split.
- **Cross-domain check** — image-classification distillation (CIFAR-100, ImageNet) with standard
  teacher/student CNN pairs, to test whether a logit-level objective transfers beyond text.
- Protocol: AdamW, cosine learning-rate schedule, batched token-level training; baselines run on the
  identical loop so that differences reflect the objective alone.

## Code framework

The objective plugs into a distillation training harness that already exists: a forward pass produces
student and teacher logits over the same (prompt + completion) tokens; a mask marks which positions
count; the per-token divergence is reduced to a scalar that becomes the training loss; the optimizer
steps the student. Everything except the divergence body is settled. The single empty slot is the
loss module itself: given the two logit tensors, the token mask, and the trainer object that ordinary
Lightning loss modules receive, return the scalar to minimize and any scalar diagnostics.

```python
from typing import Dict, Optional, Union
import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch import LightningModule


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module: LightningModule,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError


def forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    teacher_probs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class CandidateDistilLoss(DistilLoss):
    def compute_loss(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: the distillation objective we will design.
        raise NotImplementedError

    def forward(
        self,
        lightning_module: LightningModule,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        loss = self.compute_loss(logits, teacher_logits, mask)
        return {"distil_loss": loss}
```

The harness supplies the two logit tensors, the mask, and the trainer object; `compute_loss` is the
empty slot.
