The whole task lives in one function — everything from the GPT-2 Medium architecture to the FineWeb data to the AdamW schedule is frozen, and only the map from logits to scalar loss is mine to design. So the right place to start is to stare at the objective I already have. Plain next-token cross-entropy, for one position with logits $z$ over the vocabulary and target index $y$, is

$$-\log \operatorname{softmax}(z)_y = -z_y + \log \sum_j \exp(z_j).$$

This is maximum likelihood and it is correct; what bothers me is *where its optimum sits*. To drive it to zero I need $\operatorname{softmax}(z)_y \to 1$, which happens only as the gap $z_y - \max_{j\neq y} z_j$ runs off to $+\infty$. There is no finite $z$ that minimizes it. On data that is near-separable at the margin — and token-level language data effectively is, because for most contexts one next token is overwhelmingly likely — the loss is literally a standing instruction to grow the correct logit's lead without bound. The gradient $\partial\ell/\partial z_k = p_k - \mathbf{1}[k=y]$ is bounded in $[-1,1]$, so every step is gentle, but its target is at infinity, and over thirteen thousand iterations a bounded gradient pointed at infinity walks the logits steadily upward. That is over-confidence by construction: the model learns to slam probability mass onto the observed next token of every context, which overfits and makes it rigid — once a huge gap is set, a bounded gradient can only chip at a gap the loss keeps trying to make infinite. This is exactly what Szegedy and co-authors (2016) named when regularizing the Inception net.

There are three distinct handles I could pull. I could attack the absolute logit *level* by penalizing the log-partition; I could attack the logit *values* by squashing them through a bounded map; or I could attack the *target* and stop asking for probability one in the first place. I take the target first because it is the cheapest, the best understood, and its failure mode will point cleanly at the next rung: it is the most surgical statement of "the optimum is at infinity, so move the optimum to a finite place," a one-line change, and the right baseline for the harder numerical interventions to beat.

I propose **label smoothing**. If the problem is that the target lives at infinity, then do not put the target at infinity: give every token a small floor of target probability. Bleed a fraction $\varepsilon$ of the mass off the one-hot onto a fixed prior $u$ over the vocabulary,

$$q'(k) = (1-\varepsilon)\,\delta_{k,y} + \varepsilon\,u(k),$$

and absent any prior knowledge $u$ is uniform, $u(k) = 1/V$, so $q'(k) = (1-\varepsilon)\,\delta_{k,y} + \varepsilon/V$. This kills the runaway exactly: every entry of $q'$ is now at least $\varepsilon/V > 0$, so if $z_y$ tried to escape to $+\infty$ — driving $p_y \to 1$ and $p_k \to 0$ for $k \neq y$ — the cross-entropy $-\sum_k q'(k)\log p_k$ would blow up on the wrong-class terms, because $q'(k) = \varepsilon/V$ is positive while $\log p_k \to -\infty$. An infinite logit gap is now infinitely *expensive* rather than free, and the optimum sits at a finite, $\varepsilon$-controlled configuration with optimal predicted probabilities $1-\varepsilon$ on the true token and $\varepsilon/(V-1)$ on each other.

The structure is what tells me what I am really doing. Cross-entropy is linear in the target, so

$$H(q',p) = -\sum_k q'(k)\log p_k = (1-\varepsilon)\,H(q,p) + \varepsilon\,H(u,p):$$

ordinary hard-label cross-entropy downweighted by $(1-\varepsilon)$, plus an $\varepsilon$-weighted term pulling the prediction toward the prior. Since $H(u,p) = D_{\mathrm{KL}}(u\,\|\,p) + H(u)$ and $H(u)$ is constant, that second term is, up to a constant, a penalty on how far $p$ has drifted from uniform, with relative weight $\varepsilon/(1-\varepsilon)$ — a "stay a bit humble" regularizer. With $u$ uniform, $H(u,p) = \operatorname{mean}_k(-\log p_k)$, so I never have to materialize the smoothed target vector at all.

Two task-local choices are where this diverges from the generic recipe, and I have to get the first one right or I am cheating. The validation metric is the *honest* modeling cross-entropy on FineWeb — plain $-\log p_y$, no smoothing. Label smoothing deliberately fits the model to a softened distribution, so a model trained to put $1-\varepsilon$ on the true token is a worse density estimator under the true one-hot likelihood; that is an accepted trade in the original setting, where smoothing buys calibration, but *here I am graded on that likelihood*. If I left smoothing on at evaluation, the reported number would be cross-entropy against the softened target, not the data — exactly the "do not lower the reported loss by distorting the distribution" violation the contract forbids. So smoothing is applied **only during training**: when gradients are enabled I smooth, and under the no-grad eval pass I fall back to standard cross-entropy, so `val_loss` is computed against the true targets and stays comparable across the ladder. The second choice is the coefficient. The original Inception/Transformer setting uses $\varepsilon = 0.1$; this is a *short* run — thirteen thousand iterations, a tiny fraction of a full schedule — on a ~50k vocabulary, and on a short run the model never gets far enough into the over-confident regime for a heavy regularizer to pay for itself; an aggressive $0.1$ mostly biases the objective off the data without the long-run overfitting it would otherwise prevent. So $\varepsilon = 0.05$: enough of a floor to take the sharpest edge off the runaway, light enough that the $(1-\varepsilon)$ reweighting of the true-likelihood term is barely perturbed.

It is worth being clear about what this floor *cannot* reach, because that is the entire reason to run it first. Smoothing acts on the target, and through it caps the *gap* between the true-class logit and the rest — the gauge-invariant difference the softmax actually sees. It pointedly does not touch the absolute *level*: a model can sit at logits $(1000, 990, 990, \dots)$ with exactly the smoothed-optimal gaps while being numerically enormous, because cross-entropy and its smoothed cousin are both invariant to a uniform shift of all logits. So the overall magnitude — the gauge — stays completely free, and this run trains in bfloat16, where large logit magnitudes are precisely where the softmax's exponential becomes unfaithful. If this rung comes in *behind* a method that attacks the level or the values directly, the diagnosis is already written: smoothing pulled on the gauge-invariant handle and left untouched the magnitude the bfloat16 softmax cares about, which says the next rung should stop attacking the target and start attacking the level.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 1: label smoothing (eps=0.05, training only)
def compute_loss(logits, targets):
    """Cross-entropy with label smoothing (eps=0.05) during training only.

    Label smoothing is applied only when gradients are enabled (training).
    During evaluation (@torch.no_grad()), standard cross-entropy is used
    so that val_loss remains comparable across methods.
    """
    smoothing = 0.05 if torch.is_grad_enabled() else 0.0
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=-1, label_smoothing=smoothing
    )


# training-setup hook left at the default — only the loss changed:
CONFIG_OVERRIDES = {}
```
