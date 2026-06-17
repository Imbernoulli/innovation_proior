# Bad Teacher (incompetent-teacher unlearning), distilled

Bad Teacher is an approximate machine-unlearning method that removes a forget set `D_f`'s influence from
an already-trained deep classifier by *selective knowledge distillation* from two frozen teachers into a
student. On retain data the student is distilled toward a frozen copy of the original model (the
**competent** teacher); on forget data it is distilled toward a frozen *randomly-initialized* same-arch
model (the **incompetent** teacher). The student is initialized to the original weights, so it keeps its
utility on `D_r` while its behavior on `D_f` decays to that of an untrained network — uncertain, not
confidently wrong. It needs no retrained model, no second-order information, no auxiliary trained model,
and places no constraint on how the original model was trained; it supports class, multi-class, sub-class,
and random-subset forgetting.

## Problem it solves

Given a model trained on `D_c = D_r ∪ D_f`, produce — cheaply, starting from the trained weights — a
model that behaves like the one retrained on `D_r` alone: full accuracy on `D_r`, and *generalization-level*
(uncertain) behavior on `D_f` rather than exactly-zero / confidently-wrong behavior, which would itself
leak membership information.

## Key idea

Distillation makes a student copy whatever distribution a teacher emits, regardless of whether the teacher
is good. So use two teachers and route each sample by an unlearning label `l_u` (1 = forget, 0 = retain):

- **Competent teacher `T_s`** = frozen copy of the original trained model → the retain target.
- **Incompetent teacher `T_d`** = frozen, freshly random-initialized copy of the same architecture
  (diffuse, untrained outputs) → the forget target.
- **Student `S`** = initialized to the original weights; the model being unlearned.

Distilling `S` toward `T_d` on `D_f` pulls those predictions to untrained-level randomness (the privacy-safe
"never learned this" regime) instead of to a confident wrong class; distilling toward `T_s` on a retained
subset in the mixed unlearning loader holds utility and prevents the random signal from corroding nearby
retain behavior.

## Objective

With unlearning label `l_u ∈ {0,1}` per sample:

```
L(x, l_u) = (1 - l_u) · KL( T_s(x) ‖ S(x) )  +  l_u · KL( T_d(x) ‖ S(x) ).
```

Because `l_u` is a hard 0/1 selector, this equals one KL from a per-sample mixed teacher target to the
student. With temperature `T` (default `T = 1`; no hard-label term is mixed in, so no `T^2` multiplier is
used):

```
t_s = softmax(T_s_logits / T),   t_d = softmax(T_d_logits / T),   s = softmax(S_logits / T)
target = l_u · t_d + (1 - l_u) · t_s
L = KL( target ‖ s ).
```

In PyTorch this is `F.kl_div(log_softmax(S_logits / T), target)`: the first argument is the student's
log-probabilities, the second is the teacher-mixture probability target, so the mathematical direction is
`KL(target ‖ S)`. A per-sample mathematical average would use `reduction="batchmean"`; the reference
`UnlearnerLoss` leaves PyTorch's default reduction, `mean`, which divides over samples and classes and
therefore rescales the same gradient direction by the class count. The teachers are forwarded under
`no_grad` and set to `eval()`; only `S` receives optimizer updates. The reference routine uses a shuffled
mixed dataset containing all forget examples plus a retained subset, accepts Adam or a supplied optimizer,
sets `KL_temperature = 1`, and uses no `T^2` multiplier.

## Evaluation aid — ZRF (Zero Retrain Forgetting)

A retrained-model-free check of *how random* the unlearned model `M` is on `D_f`, using the incompetent
teacher as the random-behavior reference and Jensen–Shannon divergence
`JS(p,q) = ½ KL(p‖m) + ½ KL(q‖m)`, `m=(p+q)/2`:

```
ZRF = 1 - (1/n_f) Σ_{i=1}^{n_f} JS( M(x_i), T_d(x_i) ) .
```

ZRF near 1 means the model's forget-set outputs match the incompetent teacher; lower values mean a stronger
model-specific pattern remains. The target is the value a model that never saw `D_f` would give — proxied by
ZRF on a held-out test set. This is a diagnostic, not part of the update. To implement the displayed
standard JS in PyTorch, `KL(p‖m)` uses `F.kl_div(log_m, p)`. The public helper instead calls
`F.kl_div(log_p, m)` with default reduction, so it is a symmetric reverse-KL proxy with the reference code's
scale, not a reason to flip the displayed formula.

## Working code

Filling the `candidate_unlearning_loss` slot of the harness with the dual-teacher KL, and the routine that
drives it:

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
            return self.forget_data[index][0], 1
        return self.retain_data[index - self.forget_len][0], 0


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

Implementation note: the reference `UnlearnerLoss` uses the same `softmax` / `log_softmax` placement and
the same PyTorch argument convention shown here. If `reduction="batchmean"` is substituted, the formula
becomes the mathematically exact per-sample KL average, but it no longer matches the reference loss scale.
The public notebook instantiates the unlearning teacher separately as an untrained same-output-space model,
loads the student from the original trained weights, and passes `KL_temperature = 1`.
