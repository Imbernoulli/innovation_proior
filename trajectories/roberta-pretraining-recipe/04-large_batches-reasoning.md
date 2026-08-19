The four-way format comparison settled the question I actually cared about: sentence-pair+NSP, which
keeps the NSP loss but shrinks segments to single sentences, is the weakest of the four configurations
on every downstream task despite retaining NSP — 88.7/76.2 SQuAD, 82.9 MNLI-m, 92.1 SST-2, 63.0 RACE,
clearly behind segment-pair+NSP's 90.4/78.7, 84.0, 92.9, 64.2. That is the signature of span length
mattering, not the loss: chopping segments down to single sentences starves the model of exactly the
long-range contiguous text it needs, regardless of whether an auxiliary classifier is attached. Meanwhile
both NSP-free formats — full-sentences at 90.4/79.1, 84.7, 92.5, 64.8, and doc-sentences at 90.6/79.7,
84.7, 92.7, 65.6 — meet or beat segment-pair+NSP on every column except SST-2, where both are a few
tenths behind. So the loss was never the load-bearing part of the original recipe; the long, mostly-
uninterrupted spans were. That resolves the tension I flagged going in: the concurrent work questioning
NSP's necessity was right, and the earlier finding that removing it hurt performance most likely reflects
the confound I suspected — dropping the loss while keeping the two-segment structure, leaving an
unsupervised second half in the input. Between the two NSP-free formats, doc-sentences is marginally
ahead on SQuAD 2.0 and RACE and ties on the rest — but as committed before running this comparison, I'm
carrying full-sentences forward regardless, because doc-sentences' variable batch size (short documents
near the end of the corpus force a batch-size bump to keep tokens-per-batch comparable) would corrupt
exactly the controlled comparison I need for the rung I'm about to run. So: dynamic masking,
full-sentences without NSP, both locked in as defaults from here forward.

That leaves batch size, and this is a different kind of change from the two before it — not a
correction to something subtly wrong in the recipe, but a lever that trades wall-clock/hardware
efficiency directly against optimization dynamics, one I have separate reason to expect matters. Neural
machine translation has an established result that training with very large mini-batches, provided the
learning rate is scaled up appropriately to compensate for the reduced gradient-noise-per-step, can
improve *both* optimization speed and final task performance — not just training faster to the same
place, but reaching a better place. And this isn't hypothetical for masked language modeling
specifically: BERT itself has separately been shown amenable to large-batch training, with successful
runs reported at batch sizes up to 32K sequences, an order of magnitude beyond the original 256.

The reason to expect an accuracy effect and not just a speed effect is about gradient noise and the
effective step the optimizer takes. A batch of 256 sequences gives a noisier estimate of the true
gradient than a batch of 2,048 or 8,192; noisier gradients mean the optimizer's steps wander more, which
can be a form of implicit regularization at small scale but can also just mean the model spends part of
its finite step budget correcting for its own estimation noise rather than making consistent progress
toward a better minimum. Larger batches, with a correspondingly larger learning rate to keep the
effective step size in a sane range, should let each step be a more reliable descent direction — which
matters especially because I'm not adding compute, I'm redistributing it. The natural way to test this
without confounding batch size with total training compute is gradient accumulation: 256 sequences over
1,000,000 steps is, in terms of total sequences observed, equivalent to 2,048 sequences over roughly
125,000 steps, or 8,192 sequences over roughly 31,000 steps — the same number of passes through the same
amount of data, just packaged into fewer, larger updates. That equivalence is exactly what I need to
isolate batch size as the one variable under test, rather than accidentally also testing "more total
compute."

There's a second reason to want this rung specifically, beyond whatever it does to end-task numbers:
larger batches parallelize better across many workers, whether or not I have that scale of hardware
sitting idle right now, and every remaining rung in this study is heading toward much larger pretraining
runs — more data, more steps. If a large batch size is going to be part of the final recipe anyway, for
efficiency reasons independent of accuracy, I want to establish now, cheaply, at the current 16GB/BERT-
base scale, whether it also costs or helps accuracy, rather than discovering a batch-size confound later
when it's entangled with a much more expensive data-scaling or step-scaling change.

I want three points on this curve, not just a before/after, because I don't yet know whether the
relationship between batch size and downstream quality is monotonic, has an interior optimum, or is
roughly flat once the learning rate is properly retuned for each setting. 256 is the anchor — the
configuration already running as rungs 1 through 3. 2,048 (2K) is a moderate jump, roughly the middle of
the log-scale range between the original setting and the largest batch sizes reported as workable for
BERT elsewhere. 8,192 (8K) pushes further, still comfortably inside the up-to-32K range that's been
shown not to break training outright, and is the batch size I'd actually want to standardize on for the
data- and step-scaling rungs later if it holds up — training at 8K sequences per step means far fewer
total optimizer steps are needed to cover a given amount of data, which matters directly once the corpus
size grows well past 16GB. At each of the three points the learning rate is tuned separately, since a
fixed learning rate almost certainly isn't right across a 32x range of batch sizes — the whole
NMT-literature argument for large-batch training is contingent on scaling the learning rate to match, not
a claim that batch size alone, at a fixed LR, helps.

Alongside the three downstream numbers (MNLI-m and SST-2 dev accuracy, matching the metrics already used
in the format comparison, kept for direct comparability across rungs), I want to track held-out
perplexity on the masked-language-modeling objective itself at each batch size. Perplexity is a purer
read on how well the optimization dynamics are working, decoupled from whatever noise finetuning adds on
top — if perplexity and downstream accuracy move together across the three settings, that's a strong
signal the batch-size effect (whatever it turns out to be) is really an optimization-quality effect, not
finetuning noise dressed up as a pretraining result. If they diverge, that itself would be informative
about whether pretraining-objective quality and downstream task quality are as tightly coupled as I've
been assuming they are throughout this study.

So: hold dynamic masking and full-sentences-without-NSP fixed, hold total training compute
(sequences-times-steps) fixed at the level equivalent to the original 256×1,000,000 configuration, and
sweep batch size across 256, 2,048, and 8,192 sequences, retuning the learning rate at each point;
measure held-out perplexity plus MNLI-m and SST-2 dev accuracy at all three, and let that curve decide
whether large-batch training is a genuine free lunch here or a lever with limits.
