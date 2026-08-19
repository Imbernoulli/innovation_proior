Every large model I train runs on AdamW, and nobody chose that update rule by searching a space of
alternatives — it stuck because it works well enough and nobody has beaten it badly enough to
dislodge it. I want to actually search for a better rule, but before investing in any search
machinery of my own I should ask a cheaper question first: has anyone already searched, and does
what they found already clear the bar?

Two prior attempts at automatic optimizer discovery exist, and they fail for opposite reasons,
which is useful because the failure modes tell me what any search I build has to avoid. "Learning
to optimize" parameterizes the update rule as a small neural network trained on a handful of small
tasks — I can rule this out on inspection: a network trained on a few thousand steps of a toy task
has no mechanism to behave sensibly a hundred thousand steps into training a model orders of
magnitude larger, and I have no way to read *why* it does what it does, so I couldn't diagnose a
failure even if I saw one. I will not build on black-box learned optimizers. The second branch is
different and deserves an actual test: Neural Optimizer Search treats the update rule as a small
expression tree over the gradient g and a bias-corrected momentum m, searched by RL or Monte Carlo
sampling. This is a real search over real executable programs, and it produced two named,
genuinely novel results — PowerSign, which scales the gradient by e raised to ±1 depending on
whether sign(g) and sign(m) agree, and AddSign, which scales it by 2 when they agree and by 0
(skips the step) when they disagree. Neither resembles Adam, NAdam, or plain signSGD, so I owe
this branch a real check rather than an armchair dismissal.

I propose plugging both, exactly as published, into my actual target recipe — ViT-S/16 and
ViT-B/16 trained from scratch on ImageNet with RandAugment and Mixup, each rule's own learning
rate and weight decay tuned on a log scale the same way AdamW's were — and reading all three
columns (ImageNet, ReaL, V2) at both scales against the AdamW bar. If either rule clears it, I
inherit a validated result and save myself the cost of building a search of my own. My actual
worry about this search space, stated as a falsifiable prediction rather than an assumption: its
operands are fixed to {g, m}, and m itself is handed to the tree already computed by a fixed,
unsearched EMA rule — the search only ever controls how a gradient gets *combined with* a fixed
memory, never how the memory itself is tracked. If the real design freedom in an update rule lives
in how the buffer is tracked rather than just how it's combined with the instantaneous gradient,
this space is structurally capped below whatever a freer search could reach, no matter how well it
searches within its box. That's a prediction, not a conclusion — the training numbers either
support it or don't. Practically, this check costs almost nothing: two already-specified rules,
tuned the same way every other entrant is tuned, no new machinery to build. Building a program
search of my own — representation, mutation, pruning, meta-validation — is a real investment I
want to make only once I know it's necessary, and this is the cheapest experiment that could make
it unnecessary.

```
PowerSign:  update = e^{sign(g) * sign(m)} * g       # alpha = e, f(t) = 1 (default)
AddSign:    update = (1 + sign(g) * sign(m)) * g      # alpha = 1, f(t) = 1 (default)
# m: bias-corrected exponential moving average of g (standard Adam-style momentum)
```
