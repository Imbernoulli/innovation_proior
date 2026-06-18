**Problem.** PGD is the principled strongest single-loss, single-run, fixed-step first-order attack,
and it plateaued on VGG (0.720 / 0.736). Two failure modes survive it and are invisible from inside a
CE-PGD run: a fixed step size that cannot be both coarse (exploration) and fine (refinement), and a
cross-entropy loss whose scale degree of freedom makes its gradient vanish on overconfident,
well-classified points — exactly the VGG survivors — so a decision-identical rescaling can blind it.

**Key idea.** AutoAttack (standard): the worst case over a diverse, parameter-free ensemble. Two new
pieces fix PGD's failure modes — APGD makes the step size adaptive (start at `2·eps`, halve at
progress-based checkpoints, restart from the best point, heavy-ball blend) so its only knob is the
budget; and the DLR loss `−(z_y − max_{i≠y} z_i)/(z_{π1} − z_{π3})` is a ratio of logit differences,
shift- and positive-scale-invariant like the decision itself, so it keeps a gradient where saturated CE
goes flat. Two complementary existing attacks add diversity: targeted FAB (boundary linearization,
norm-minimizing) and Square Attack (gradient-free score-based search, the gradient-masking backstop). A
sample is robust only if *none* of {APGD-CE, targeted APGD-DLR, targeted FAB, Square} flips it.

**Why it is the endpoint.** It is not a better CE climber but a *reliable* one: it cannot be inflated
by a loss-scale quirk or a single attack's blind spot. On these undefended models most VGG survivors
are architectural `L_inf` robustness, so the expected gain over PGD is small — the value is that the
worst-case-over-ensemble number is trustworthy. Nothing clearly stronger and published fits this edit
surface, so the ladder stops here.

**Scaffold grounding.** The full ensemble lives in `torchattacks`; the task's edit fills `run_attack`
with the single standard-ensemble call, not a reimplementation. `n_classes` is passed (it sets the
targeted members' class sweep: 9 of 99 wrong classes for CIFAR-100); `seed` is read from the
environment for reproducibility under harness seed 42; `device` is unused (the package places tensors).

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import os
    import torchattacks

    _ = device
    model.eval()
    attack = torchattacks.AutoAttack(
        model,
        norm="Linf",
        eps=eps,
        version="standard",
        n_classes=n_classes,
        seed=int(os.environ.get("SEED", "42")),
        verbose=False,
    )
    return attack(images, labels)
```
