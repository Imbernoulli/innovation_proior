The confidence-greedy run told me exactly where the selection signal is leaking, and it told me in
numbers. On math it landed 0.316 accuracy, on HumanEval 0.3659 — respectable, clearly above a blind
random order, so using the model's own certainty at all was the right move. But two things in the trace
are sharp. First, `avg_steps` is pinned at the full budget everywhere: 256 on math, 256 on HumanEval, 224
on the single-block text run. That is exactly the structural feature I flagged — it commits precisely the
schedule's `k` per step and never more, so it spends every forward pass it is allowed and the efficiency
metric sits at its worst possible value. Second, and the reason I am here, the open-ended text numbers are
poor: gen_ppl 170.6, MAUVE 0.0321, with a high bigram entropy 6.41 that I do not read as "diverse" so
much as "incoherent" — entropy without MAUVE is just noise that happens to use many word types. So the
quality problem is concentrated where it should be, and the diagnosis I wrote at the close of the floor
step is the one to act on: the score `max_v p(v)` is a function of a single number, the winning mass, and
it cannot tell a decisive top token from one in a dead heat with its runner-up. The 170.6 perplexity on a
single parallel block is the model freezing near-tie tokens early, in any order, and then conditioning
the whole continuation on those frozen coin flips. The information that would separate the safe commit
from the dangerous one is sitting in the logits and confidence-greedy throws it away. So this step is not
a new family of method — it is the *minimal* repair of that one leak.

Let me re-derive the failure precisely so the fix is forced rather than guessed. Picture two masked
positions on the same step. At position A the top token is at probability 0.45 and everything else is
scattered tiny — the runner-up is down at 0.02. At position B the top token is also at 0.45, but the
second-place token is at 0.44 — a dead heat between two candidates with the rest of the mass negligible.
The confidence-greedy score, `max_v p(v)`, reads both as 0.45 and calls them equally safe. They are not
remotely equally safe. At A the model is decisively committed: one answer, a more-than-twentyfold margin
over the next. At B the model is genuinely torn between two near-identical options, and because the
harness writes the argmax and never revisits a committed token, whichever I freeze kills an almost-equally-
good alternative permanently. The entire premise of "commit where the model is sure" was to avoid exactly
B, and `max_v p(v)` walks straight into it, because 0.45 does not know the difference between "0.45 versus
0.02" and "0.45 versus 0.44." That is the 170.6-perplexity story made concrete: on the single text block,
the first step has almost no context, lots of positions are near-ties at modest probability, and
confidence-greedy commits a budget's worth of them anyway, freezing coin flips into the bidirectional
context that the rest of the 224-token continuation has to denoise around. The early near-tie commitments
are the early mistakes, and they are most expensive in the fully-parallel regime where the block is the
whole region and there is no left-to-right backbone to lean on.

So what is the minimal thing I have to add back? Not the whole distribution — just the piece that A and B
differ on, which is the second-place probability. The honest measure of "is the top token clearly the
winner here, or is it in a fight" is the *gap* between first and second:

  margin = P(top1) - P(top2),

the top-1 probability minus the top-2 probability at each position. Read it on my two cases: A gets
`0.45 - 0.02 = 0.43`, B gets `0.45 - 0.44 = 0.01`. Now A scores forty-three times higher than B, which is
exactly the ranking I wanted — commit A, leave B masked until its context firms up and the tie resolves.
A large gap means the top token dominates its nearest rival, so freezing it costs me nothing I would
plausibly have wanted; a small gap means a near-tie, the one case where my irreversible commit is a
gamble. So the scalar to maximize for "safest to commit" is the margin, and the extra cost is essentially
nothing: one descending sort gives me the first two probabilities I need for the score, while the token
assignment stays the ordinary argmax. Crucially, this is a pure *selection-signal* swap — I keep the
budget, the block walk, the carry-over, the eligibility masking, and the argmax token exactly as
confidence-greedy had them, and change only the number I rank by. That is why I expect it to be a small,
clean delta on the same scaffold rather than a different decoder.

I want to make sure I am not reinventing a thing with a known failure mode I am forgetting, so let me
place this gap. It is precisely the quantity active learning calls *margin*: rank items by
`P(y_1 | x) - P(y_2 | x)`, the spread between the two most probable classes, introduced exactly to fix
the shortcoming of the least-confident criterion, which looks only at the top class and "throws away
information about the remaining label distribution." That critique is word-for-word my A-versus-B
complaint about confidence-greedy's `max_v p(v)`, which is the least-confident measure read in the
certainty direction. So the margin is the standard repair for the disease I diagnosed. One thing I have
to be careful about is *direction*: in active learning you query the *smallest* margin (most ambiguous,
most worth a human label). I want the opposite — I am committing the positions I am already certain about
and leaving the uncertain ones alone — so I take the *largest* margin. Same scalar, opposite end. It is
the certainty reading of best-versus-second-best, keeping the bests rather than querying the worsts. This
is also the `topk_margin` selection option in Dream's diffusion-LM decoding (Ye, Xie et al. 2025): rank
masked positions by top1-minus-top2 and decode the largest-margin ones first.

Should I instead use the full distribution and go for entropy, which uses *everything*, not just the top
two? Entropy is `-Σ_v p(v) log p(v)`; low entropy means peaked, peaked means confident, so I would unmask
the lowest-entropy positions. In a tiny label space that is a reasonable confidence proxy because there
is not much tail to account for. But my vocabulary is not tiny — it is tens of thousands of classes. Look
at the entropy sum in that regime. The overwhelming majority of the terms are tail tokens at near-zero
probability, each contributing a tiny `p log p`, and there are tens of thousands of them. That huge pile
of small contributions can shift the entropy as much as the contest at the very top does. Two positions
could have identical top-two structure — same `P(top1)`, same `P(top2)` — and entropy could still rank
them differently purely because of how the negligible mass is smeared across the long tail, which is
noise I do not care about. The thing that actually predicts whether my commit is correct is whether the
top token clearly beats its nearest rival; entropy dilutes that signal with tail bookkeeping. This is the
many-class argument for best-versus-second-best: when there are lots of classes the spread over the tail
is uninformative relative to the gap between the two front-runners, so the two-best gap is the robust
certainty signal and full entropy is the noisy one. So margin is the cheap, robust middle: it adds back
exactly the one piece `max_v p(v)` was missing — the runner-up — and ignores exactly the tail noise
entropy gets distracted by.

Now the token assignment, which I should re-examine rather than assume. Once I have decided a position is
decisive — a large margin — the natural commit is its most probable token, the argmax. On a position I
have specifically judged safe I do not want to inject variance; the whole reason I chose it is that its
top token dominates, so I take it. This is the same greedy token confidence-greedy used, and I keep it for
the same reason: on the accuracy tasks there is one correct answer and the mode is what I want.

Let me make the step concrete in this scaffold, because the abstraction has to become tensor ops in the
one empty slot. The model gives me logits over the whole sequence. I want probabilities, and I want them
in enough precision that the top-two gap of two close masses is not crushed by float rounding, so I
softmax in float64 — the same precision choice the floor made, now load-bearing because the margin is a
*difference* of two masses and a difference is exactly where low precision bites. From `p = softmax`
I take `x0 = argmax(p)` for the token. For the score I sort `p` descending along the vocabulary axis and
read the first two columns: `margin = sorted_p[..., 0] - sorted_p[..., 1]`, which is `P(top1) - P(top2)`
at every position in one shot. The selection is identical in shape to the floor: only masked positions in
the current block are eligible, so I set the margin of every ineligible position to `-inf` and take the
top-`k` by margin where `k` is the schedule's budget for this step, then write `x0` there. Everything
else stays masked for next iteration. One forward pass per step, one sort, one top-`k`, one masked write
— the only line that changed from confidence-greedy is the score: `margin` from the two largest sorted
probabilities instead of `conf` from the single largest. The block structure falls out unchanged: one
block when `block_length == gen_length` (fully parallel, the text regime), `gen_length / block_length`
blocks walked left to right otherwise (semi-autoregressive, the accuracy regime), with eligibility
restricting the top-`k` to the active block. The full scaffold module is in the answer.

So reading the floor's numbers, here is what I expect this to fix and where I am unsure. On open-ended text
I expect the clearest win, because that is where confidence-greedy bled: the fully-parallel single block
is exactly the regime where early near-tie commitments do the most damage, and margin is the rule that
refuses to freeze a near-tie. I expect MAUVE to climb from 0.0321 — distributional similarity is the
metric most directly improved by not poisoning the context with frozen coin flips — and I would not be
surprised if gen_ppl stays high or even rises somewhat, because margin is still a greedy one-block decoder
with no diversity injection, and lower perplexity is not the same thing as higher distributional match.
That is the tension I will watch: a margin-selected continuation can read as more *coherent to the
reference distribution* (higher MAUVE) while a fixed external scorer's perplexity moves the other way. On
the accuracy tasks I expect a small, real lift over the floor — margin should edge math up from 0.316 and
HumanEval up from 0.3659, with HumanEval the more likely to move because code has many positions that are
genuine near-ties (variable names, formatting tokens) where margin's tie-avoidance helps. And I expect
`avg_steps` to stay pinned at the full budget — 256, 256, 224 — exactly as the floor, because margin
changes *which* `k` positions I commit, not *how many*: it still spends the entire schedule, one forward
pass per slot. That is the one thing margin cannot fix, and it is already pointing at the next rung. If
the margin rung confirms that the quality gains come from being more careful about *which* tokens to
freeze, while the cost stays pinned because the *count* is fixed by a static schedule, then the move after
this is clear: make the count adaptive — commit *many* positions in one step when the model is settled and
sure, and only fall back to the schedule when it is not — so I can finally pull `avg_steps` down off the
full budget without giving back the quality the margin bought me.
