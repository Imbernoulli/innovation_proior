The four-way format comparison settled the question I actually cared about. Sentence-pair+NSP, which
keeps the NSP loss but shrinks segments to single sentences, is the weakest of the four configurations
on every downstream task despite retaining the loss — 88.7/76.2 SQuAD, 82.9 MNLI-m, 92.1 SST-2, 63.0
RACE, clearly behind segment-pair+NSP's 90.4/78.7, 84.0, 92.9, 64.2. That's the signature of span length
mattering, not the loss: chopping segments down to single sentences starves the model of exactly the
long-range contiguous text it needs, whether or not an auxiliary classifier is attached on top. Both
NSP-free formats, meanwhile, meet or beat segment-pair+NSP on nearly every column: full-sentences at
90.4/79.1, 84.7, 92.5, 64.8, doc-sentences at 90.6/79.7, 84.7, 92.7, 65.6, both slightly behind only on
SST-2. So the loss was never the load-bearing part of the original recipe — the long, mostly
uninterrupted spans were. That resolves the tension I flagged before running this: the concurrent work
questioning NSP's necessity was right, and the earlier claim that removing NSP hurt performance most
likely reflects the confound I suspected going in — dropping the loss while leaving a structurally
two-segment input whose second half no longer has any signal explaining why it's there. Between the two
NSP-free formats, doc-sentences edges ahead on SQuAD 2.0 and RACE and ties the rest, but as I committed
before this comparison even ran, I'm carrying full-sentences forward anyway — doc-sentences' variable
batch size would corrupt exactly the controlled comparison I'm about to set up. Dynamic masking and
full-sentences-without-NSP are both locked in as defaults from here on.

That leaves batch size, and it's a different kind of change from the two before it — not a correction to
something subtly wrong in the recipe, but a lever that trades hardware efficiency directly against
optimization dynamics, and one I have independent reason to expect matters. Machine translation has an
established result that training with very large mini-batches, provided the learning rate is scaled up
to compensate for the reduced per-step gradient noise, can improve *both* optimization speed and final
task quality — not just reaching the same place faster, but reaching a better place. And this isn't
hypothetical for masked language modeling specifically: BERT has separately been shown amenable to
large-batch training, with workable runs reported at batch sizes up to 32K sequences, more than two
orders of magnitude beyond the original 256.

The case for an accuracy effect, not just a speed effect, comes down to gradient noise and the
reliability of each step. A batch of 256 gives a noisier estimate of the true gradient than a batch of
2,048 or 8,192; noisier gradients mean the optimizer wanders more per step, which can act as a mild
implicit regularizer at small scale but can also just mean part of a finite step budget goes to
correcting for the optimizer's own estimation noise rather than making consistent progress. Larger
batches, with a correspondingly larger learning rate keeping the effective step size sane, should make
each step a more reliable descent direction — and this matters especially because I'm not adding
compute, I'm redistributing it. Gradient accumulation lets me hold total compute fixed while varying
batch size: 256 sequences over 1,000,000 steps is, in total sequences observed, equivalent to 2,048
sequences over roughly 125,000 steps, or 8,192 sequences over roughly 31,000 steps — same total data
exposure, different packaging. That equivalence is exactly what isolates batch size as the one variable
under test, rather than smuggling in "more total compute" alongside it.

There's a second reason to want this rung now, independent of what it does to end-task numbers: larger
batches parallelize better, and every remaining rung in this study is heading toward much larger
pretraining runs — more data, more steps. If a large batch size is going to be part of the eventual
recipe anyway for efficiency reasons, I'd rather establish now, cheaply, at the current 16GB/BERT-base
scale, whether it also helps or costs accuracy, than discover a batch-size confound later when it's
tangled up with a far more expensive data- or step-scaling change.

I want three points on this curve rather than a single before/after, because I don't yet know whether
the relationship is monotonic, flat, or has an interior optimum once the learning rate is properly
retuned at each point. 256 is the anchor, already running as rungs 1 through 3. 2,048 is a moderate
jump, roughly midway on a log scale toward the largest batch sizes reported as workable for BERT
elsewhere. 8,192 pushes further, still well inside the up-to-32K range already shown not to break
training outright, and is the batch size I'd actually want to standardize on for the data- and
step-scaling rungs ahead if it holds up — far fewer optimizer steps needed per unit of data matters
directly once corpus size grows well past 16GB. Learning rate is tuned separately at each of the three
points, since a fixed rate almost certainly isn't right across a 32x range of batch sizes — the whole
large-batch argument from the NMT literature is contingent on scaling the learning rate to match, not a
claim that batch size alone helps at a fixed rate.

Alongside MNLI-m and SST-2 dev accuracy — the same metrics used in the format comparison, kept for
direct comparability — I want held-out perplexity on the masked-language-modeling objective itself at
each batch size. Perplexity is a purer read on optimization quality, decoupled from whatever noise
finetuning adds. If perplexity and downstream accuracy move together across the three settings, that's
a strong signal that whatever batch-size effect shows up is a genuine optimization-quality effect and
not finetuning noise dressed up as a pretraining result; if they diverge, that would itself be
informative about how tightly pretraining-objective quality and downstream-task quality actually track
each other, an assumption this whole study has been leaning on.

```
batch size:    sweep {256, 2048, 8192} sequences, learning rate tuned per setting
steps:         {1,000,000, ~125,000, ~31,000} respectively — compute-matched
               (same total sequences observed across all three)
[unchanged]:   dynamic masking, full-sentences (no NSP), BERT-base architecture,
               BookCorpus+Wikipedia 16GB
```

I'll read this off held-out perplexity plus MNLI-m and SST-2 dev accuracy at all three batch sizes, and
let the curve decide whether large-batch training is close to a free lunch at this scale or a lever with
real limits.
