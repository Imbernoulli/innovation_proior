**Problem.** PowerSign/AddSign (rung 1) both fell short of AdamW at every column, both scales —
consistent with the suspicion that a fixed-operand, fixed-momentum search space caps what can be
found. Build a freer search space (arbitrary short imperative programs over AdamW's own
buffer signature, warm-started at AdamW) and see what it finds.

**Key idea.** Random sampling (2M programs) still loses to AdamW — good rules are sparse — so use
regularized evolution (tournament selection, mutate-insert/delete/modify, warm-start = AdamW),
with abstract-execution pruning (shape-checking, functional-hash caching, redundant-statement
flagging) to survive an infinite, mostly-invalid program space; evolutionary search beats both an
AdamW-hyperparameter-tuning baseline and random search even when those get 4× more compute. After
pruning (~70% of statements end up redundant; caching cuts search cost ~10×), the surviving raw
program still carries dead arithmetic, but one chain — `m2=m*m; abs_m=sqrt(m2); update=m/abs_m` —
is exactly `sign(m)` (since √(m²)=|m|). Read literally, that makes the whole rule "sign of an
accumulated momentum buffer" — the signSGD-momentum shape. Test that literal reading before
trusting the raw program's second, differently-tuned `interp` call as anything more than noise.

**Rung-2 fill.** Single EMA, then sign, at the two natural candidate rates the raw program's own
constants suggest:

```python
def train(w, g, m, lr):           # ONE extra state buffer, not two
    m = interp(g, m, beta)        # (1-beta)*g + beta*m
    update = sign(m)
    update = update + w * weight_decay
    update = update * lr
    return update, m
```

`Ablation_0.9` (β=0.9, matching ordinary momentum) and `Ablation_0.99` (β=0.99, ~10× the memory
horizon of 0.9) — same protocol as rung 1, own lr/λ tuned, lr scaled down from AdamW's since the
raw sign output is ±1 per coordinate, much larger than AdamW's m/√v step.

**Why this rung.** It's the simplest structure consistent with what survives pruning, and it's
falsifiable in a way that matters for cost: if one buffer is enough, I ship something lighter than
AdamW and I'm done; if it isn't, the raw program's insistence on two *different* interpolation
constants — which I've been provisionally reading as possibly cosmetic — becomes load-bearing.

**What to watch.** Both variants vs AdamW (78.89/84.61/66.73 ViT-S/16; 80.12/85.46/68.14 ViT-B/16)
and vs PowerSign/AddSign (77.36/83.39/65.17, 77.37/83.36/64.52; 78.95/84.76/67.46, 78.50/84.49/65.95)
— beating rung 1 would confirm freeing the momentum-tracking rule (not just tree depth) is where
the gain sits; whether either variant also clears AdamW decides whether one buffer is enough.
