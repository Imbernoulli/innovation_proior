**Problem.** Before building any new search machinery, check whether a *prior* automated
discovery already clears AdamW under the actual target protocol (ViT-S/16 & ViT-B/16, ImageNet,
RandAugment + Mixup, tuned lr/λ).

**Key idea.** Neural Optimizer Search (Bello et al. 2017) already ran a real automatic search —
RL/Monte-Carlo over expression trees with fixed operands {gradient g, bias-corrected momentum m}
and a fixed, bounded tree structure — and surfaced two named optimizers, PowerSign and AddSign.
Neither one's tree structure can touch how m *itself* is tracked; the search only ever combines a
gradient with a fixed, off-the-shelf EMA. Plug both in, unmodified, and tune lr/λ the same way
every other entrant is tuned.

**Rung-1 fill.** No new code to write — use the published formulas exactly, default constants
(f(t)=1, no internal decay):

```
PowerSign:  update = e^{sign(g) * sign(m)} * g       # alpha = e
AddSign:    update = (1 + sign(g) * sign(m)) * g      # alpha = 1
# m: bias-corrected exponential moving average of g (standard Adam-style momentum)
```

**Why this rung.** It is the cheapest possible experiment that could make a new search
unnecessary: two already-specified rules, no new machinery, exactly the tuning budget every other
entrant gets. And whichever way it comes out, it's diagnostic — a pass validates reusing a
restricted-tree search space; a fail is evidence (not proof) that the interesting freedom in an
update rule lives in how the memory buffer is tracked, which this search space never let the
search touch.

**What to watch.** All three columns (ImageNet, ReaL, V2) at both scales against AdamW's
78.89/84.61/66.73 (ViT-S/16) and 80.12/85.46/68.14 (ViT-B/16) — a rule that wins on ImageNet but
not ReaL/V2 is a weaker signal than one that wins everywhere.
