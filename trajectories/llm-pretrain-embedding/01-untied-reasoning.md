I start at the two places where this model touches its vocabulary, because that is the only place the
research question can live, and I start with the single most parameter-cheap-to-question decision in the
default: the tie. A language model meets each word twice. On the way in, the current token is effectively
a one-hot that selects a row of a matrix `U` of shape `(vocab_size, n_embd)` — one row per vocabulary
item, `n_embd` units — producing a dense vector the body consumes. On the way out, after the 24 frozen
blocks have chewed on the history and produced an activation `h ∈ R^{n_embd}`, I need one score per
vocabulary item, and the cheapest map from `h` to `vocab_size` scores is another `(vocab_size, n_embd)`
matrix `V`: logits `= V h`, then softmax, then next-token probabilities. Two matrices, identical shape,
both indexed by the same vocabulary. The default in this scaffold collapses them: `lm_head.weight` is
wired straight from `wte.weight`, so the row that embeds a word on input is the same row that scores it
on output. That is weight tying, and it is the natural floor to interrogate first — not because it is
obviously wrong, but because it is the one embedding decision the default makes silently, and the whole
ladder is about what happens at these two ends. Before I add anything (bigram statistics, value
injections), I want to know whether the two ends should even share one matrix at all. This is the
lightest possible intervention in the editable interface: it changes nothing in `forward`, only what
`get_lm_head_weight()` hands back.

"Same shape, same vocabulary index" is a seductive reason to share, but it is not "same job," and before
I commit either way I want to know what each end is actually being asked to do. Think about what I want
the *input* matrix `U` to capture. When the model reads a word, I want the body to *react* to synonyms in
nearly the same way: if "couch" and "sofa" produce almost the same perturbation of the hidden state, then
everything downstream generalizes across them for free. So the input embedding wants nearby rows for
words that play the same role *as inputs to the dynamics*. Now the *output* matrix `V`. It is literally
the weight matrix of a `vocab_size`-way softmax classifier over the next token: the score of word `k` is
`V_k · h`, an inner product of word `k`'s row with the current activation. For that classifier to be
good, I want words that are *interchangeable as continuations* — words that should be predicted in the
same contexts — to have nearby rows, so they receive similar scores against any `h`. These two notions of
"similar words" rhyme but are not identical. One is "this word pushes the state the same way when read."
The other is "this word should be scored the same way when predicted." A representation tuned for the
first is not automatically tuned for the second. So already I am suspicious of forcing one matrix to be
both, and the suspicion is about *roles*, not just parameter count.

I can make the suspicion concrete instead of hand-wavy by looking at how these matrices are actually
*trained*, because gradients do not lie about what a parameter is being pushed to become. Take the
per-step negative log-likelihood `L_t = −log p_t(o_t | i_{1:t})`, with
`p_t(o_t|·) = exp(V_{o_t} · h) / Σ_x exp(V_x · h)`, where `o_t` is the true next word and `i_t` the
current input word. Differentiate with respect to the two matrices separately, in the configuration where
they *are* separate. The loss depends on `V` directly through the logits `z_k = V_k · h`, and
softmax-cross-entropy gives `∂L_t/∂z_k = p_t(k|·) − 1{k = o_t}`, so
`∂L_t/∂V_k = (p_t(k|·) − 1{k=o_t}) · h`. Read it row by row: the true word gets a pull
`(p_t(o_t|·) − 1) h` (coefficient negative since `p ≤ 1`) that raises its score; every other word `k`
gets a push `p_t(k|·) h` that lowers its score, sized by the probability mass currently wasted on it.
Every single row of `V` gets a nonzero gradient at every step, because `p_t(k|·) > 0` for all `k`. The
output matrix is *densely* trained, all the time.

Now the input matrix `U` in the same setting. The loss depends on `U` only through `h`, and `h` depends
on `U` only through the rows actually looked up — and at this step only the current input word `i_t` was
read, so only row `U_{i_t}` is in the computational path. For every other row, `∂L_t/∂U_k = 0`. For the
row that was used, the chain rule through `h` gives
`∂L_t/∂U_{i_t} = (Σ_x p_t(x|·) V_x − V_{o_t}) · ∂h/∂U_{i_t}`. The shape is completely different from the
output side: only *one* row of `U` moves per step, and all others sit frozen. A word that is rare as an
*input* gets its input row updated only a handful of times in the entire run. So even with identical shape
and vocabulary index, the two matrices are driven by *structurally different gradient streams*: `V` dense
(every row, every step), `U` sparse (one row, the current input, per step). These are two different
training dynamics — two objects shaped by two different forces for two different purposes.

But I should not jump from "two forces" to "two matrices." I should ask honestly what happens if I *do*
tie them, `S = U = V`, since that is exactly the default I am sitting on. When `S` plays both roles, the
gradient on a row of `S` is the sum of its input-role and output-role gradients. For almost every row —
every `k ≠ i_t` — the input role contributes nothing this step, so the update is exactly the output-role
update. Only the single row `k = i_t` gets both. Stare at that special row. Its output-role term is
almost always the non-target case `p_t(i_t|·) h`, because the current input word is almost never the very
next word — immediate repetition is rare — and `p_t(i_t|·)` is tiny. So that one row is locally dominated
by its input-role term. But across the whole matrix and across time, `vocab_size − 1` rows per step are
output-only, and a fixed word is the current input only on a sparse subset of steps. So over training the
tied matrix `S` evolves to look much more like what the output embedding `V` would have been: the
output-role update is dense across rows, the input-role update is sparse across time, and the input role
simply does not get a fair say. Tying does not *blend* the two representations; it makes the shared matrix
track the output role and leaves the input role underrepresented. If I cared about the input
representation being good in its own right, tying quietly throws away some of that freedom — and this is
*precisely* the thing the default is doing, so it is the right floor to question first.

I have to be careful, though, because there is a known result that sharing input and output word
matrices is actively *bad*, and if that were a law it would settle the question. The setting where it is
bad is word2vec skip-gram: a "center" and a "context" matrix, where Goldberg & Levy argued a word is
rarely its own context, so forcing the two to be one table makes a word's self-score `U_i · U_i = ‖U_i‖²`,
and suppressing self-prediction shrinks the vector toward zero. Does that condemn tying, or is it specific
to skip-gram? The hinge is the body. In skip-gram there is essentially no body — the activation the output
scores against *is the input vector itself*, an identity body — which is why the self-prediction
pathology bites: the same vector appears on both sides of the inner product. But here there is a 24-layer
transformer between the input lookup and the output score. The activation `h` is a nonlinear function of
the *entire* history, `f(U_{i_1}, …, U_{i_t})` through the blocks, not of the current input row directly.
There is no `U_{i_t} · U_{i_t}` to collapse, because the thing `V` scores against is no longer the input
row. The body *decouples* the two ends. So the word2vec result is not a law against tying; it is a
statement about the identity-body case. With a deep body, that norm-collapse argument does not transmit,
and tying becomes *admissible but not mandatory*.

That reframes the whole thing, and it is a relief, because it means I get to choose on the merits rather
than obey a prohibition. Tying is a *constraint*: it forces `U = V`, collapsing two matrices into one.
Untying is the *unconstrained* configuration: two independent matrices, each free to follow its own
gradient stream to its own optimum. The unconstrained version is more expressive by definition — it
contains the tied solution as the special case `U = V`. Tying buys two things: it removes a
`vocab_size · n_embd` block (the largest in this model — about 51.5M parameters at
`vocab_size = 50304`, `n_embd = 1024`) and it imposes a regularizing constraint that forces every
output-classifier row to double as that word's input row. Those are benefits when parameter reduction and
overfitting control are the binding needs — which is exactly the regime small-corpus language models live
in, and exactly why tying became the default. But this task is not that regime: I have ~7.1B training
tokens against a ~355M model, an abundant-data regime where overfitting is not the binding constraint and
the saved parameters are not clearly worth the lost output-head freedom. Untying spends the
`vocab_size · n_embd` parameters so the output classifier gets its own rows — the logit map is no longer
the same table as the input lookup, and the head can carve predictive directions the input lookup has no
reason to represent — at the cost of more parameters to regularize. In a data-rich pretraining run, that
is the trade I expect to pay off, and it is the most conservative, lowest-risk place to start the ladder:
one decision, the relationship between the two word matrices, changed.

The one detail I must get right is initialization of the newly separated output matrix, because the
harness wires `lm_head.weight = embedding.get_lm_head_weight()` and then trains it. Picture step zero. If
I initialize the separate output matrix at the usual small scale (std 0.02 for the GPT-2 family), the
first logits `V h` are small random numbers — a slightly random initial next-token distribution that the
first updates must spend effort flattening. There is a cleaner start. If I initialize the output matrix to
*zero*, the initial logits are exactly 0 for every word, the softmax is exactly uniform, and the initial
loss is exactly `ln(vocab_size)` — the maximum-entropy, "I know nothing yet" baseline, with no spurious
logits to undo. From there the output rows grow from zero under their dense per-step gradient
(`(p − 1) h` on the true word, `p_t(k|·) h` on the rest), which is well behaved. So I zero-init the
separate output projection and keep `wte` at the standard `normal(0, 0.02)`, so the slow-learning
(sparse-gradient) input pathway carries signal from step 1, while the fast-learning (dense-gradient)
output pathway grows from a clean uniform start. The asymmetry in the inits mirrors the asymmetry in the
gradient streams I derived.

Everything else stays exactly as the default. Untying is an intervention on the *relationship between the
two word matrices*, nothing more: the learned absolute position embedding stays added to the token
embedding in `forward` and stays excluded from the reported parameter count, so this cannot be a back
door to capacity through positions. There is no per-layer residual here, so I do not define
`get_value_embed` (the harness hook stays inert). The output `Linear` keeps no bias, matching the fixed
head; the bias was never part of the tying question. Concretely, the only edit to the default fill is:
add a separate, zero-initialized `_lm_head_weight` parameter of shape `(vocab_size, n_embd)` in
`__init__`, and return *that* from `get_lm_head_weight()` instead of `wte.weight`. The full scaffold
module is in the answer.

This is where I plant the floor for the ladder, so let me say what I am watching and what would falsify
the bet. The bet is that in an abundant-data, large-model regime, output-specific classifier capacity is
worth more than the tied regularizer, so untying should *lower* validation cross-entropy relative to the
tied default — a modest but real drop, since this is a capacity/regularization trade, not a new signal.
The risk is the opposite: if untying mostly added 51.5M lightly-constrained parameters that the data does
not discipline, validation loss could move little or even worsen, and downstream zero-shot accuracy would
be the tell that the extra head capacity is noise rather than signal. I expect the perplexities
(WikiText-2, LAMBADA) to track validation loss, and the downstream accuracies (ARC-Easy, HellaSwag) to be
roughly flat at this rung — untying is a representational decoupling, not an injection of new context, so
I would not expect a large downstream jump. Whatever the precise numbers, this rung's job is to settle
the *tie-or-not* question cleanly so the next rungs can build on a decoupled, well-understood base; if the
untied model only matches or barely beats tied, that itself is the diagnosis that the next move must
*add* signal (local context or per-layer injection) rather than merely *free* existing parameters.
