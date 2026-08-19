The combined recipe, at a tenth of BERT-large's original step count and no additional data, already
clears every one of BERT-large's published numbers — SQuAD 93.6/87.3 against 90.9/81.8, MNLI-m 89.0
against 86.6, SST-2 95.3 against 93.7. That's a substantial gap to open with 900,000 fewer steps and the
same 16GB of raw text, and it settles what rung 5 was designed to answer: the individual recipe
corrections compound, and BERT-large's original training wasn't extracting anywhere close to what this
recipe can pull from the same corpus in a fraction of the steps. But it also reframes the question I have
left. If a corrected recipe this efficient is already ahead of the fully-trained original at 100K steps,
the natural next question is whether the remaining headroom — how much better this recipe can still get
— is a *data* ceiling, a *duration* ceiling, or both. I want to test those as genuinely separate levers,
for the same reason I insisted on isolating batch size from total compute earlier: pull data and steps
together and, if the numbers move, I won't know which one did the work — and I'll have spent the much
larger compute of both changes to answer only one question.

Data is the cheaper lever to pull first, in the sense that it doesn't require committing to a training
duration yet — I can hold steps fixed at exactly what rung 5 used, 100,000, and change only how much and
how varied the pretraining text is. There's independent reason to expect this matters. Work outside this
recipe has already shown that simply increasing pretraining data size, holding everything else fixed,
improves downstream performance — a general pretraining finding, not specific to this architecture or
objective. Several other groups building on masked or autoregressive pretraining have separately moved
to datasets substantially larger and more diverse than the original 16GB corpus, though not all of those
are obtainable, which limits how directly their results compare to anything reproducible here. The
external permutation-based model I've been using as a calibration reference throughout this study, for
instance, is trained on nearly ten times more data than original BERT and sees roughly four times as
many total sequences once its batch size and step count are accounted for — so any gap between that
model and this recipe, measured so far only at the original 16GB scale, currently conflates an
architecture/objective difference with a scale difference I haven't controlled for. I can't resolve that
conflation without training on comparable data volume myself.

So the goal for this rung is to gather as much English-language pretraining text as practically
available, across genuinely varied sources rather than one very large source, and add it to the existing
16GB corpus, while holding architecture, masking, input format, batch size, vocabulary, and step count
exactly where rung 5 left them. Three additional corpora, each chosen for a different reason. A
news-crawl corpus built from the English portion of a broad web crawl of news articles across several
years of publication — a journalistic, current-events register distinct from Wikipedia's reference prose
or BookCorpus's narrative fiction, contributing tens of gigabytes after filtering, a meaningful fraction
of the total on its own. An open-source recreation of a large web-text corpus built from URLs shared on
a social platform and filtered by community upvotes — a proxy for curated general-interest web content,
its own register again. And a smaller corpus filtered specifically to match the narrative style of a set
of common-sense reasoning benchmarks, included less for raw size and more because story-structured text
is a genre the other three sources underrepresent — diversity of register is a variable I want to vary
alongside raw byte count, not fold into it. Together with the original 16GB this totals a large multiple
of the starting corpus, roughly comparable in order of magnitude to what other groups have used, though
gathered independently.

I want to be honest about a limitation I can't fully engineer around: this rung conflates data *size*
and data *diversity*, since the added corpora are both larger in aggregate and drawn from different
domains than the original two sources. A cleaner experiment would hold total byte count fixed while
varying only source diversity, or the reverse — but that would mean either discarding usable text to hit
a byte budget or finding same-domain data at a scale that isn't available, and I don't think either
substitution is worth what I'd have to sacrifice to get it. So whatever this rung shows will be a
combined effect of more and more varied data together, not cleanly attributable to one or the other —
worth keeping in mind when interpreting the result, not a reason to hold off on running it.

The prediction I want to state plainly, without presuming its size: given the general external finding
that more pretraining data helps, and given how efficiently rung 5 already extracted signal from a fixed
16GB corpus, I expect performance to improve further on SQuAD, MNLI-m, and SST-2 relative to rung 5,
holding steps fixed at 100,000. Whether that improvement turns out small — the recipe corrections already
captured most of the available headroom, data volume a second-order effect at this step count — or large
— BERT-large was significantly data-starved on top of being duration-starved, and the two sources of
gain are close to independent and additive — is exactly what this rung is built to distinguish. And the
answer shapes how much is left for the next rung, which holds data fixed here and extends training
duration instead.

```
data:          BookCorpus + English Wikipedia (16GB) + news-crawl corpus (~76GB
               after filtering) + open web-text recreation (~38GB) +
               narrative-style crawl (~31GB) -- ~160GB total
[unchanged]:   combined recipe (dynamic masking, full-sentences/no-NSP, 8K batch,
               byte-level BPE), BERT-large architecture, 100,000 steps
```

I'll evaluate SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, and SST-2 accuracy against rung 5's own numbers,
holding step count fixed so any shift is attributable to data alone.
