The combined recipe, at a tenth of BERT-large's original step count and no additional data, already
clears every one of BERT-large's published numbers: SQuAD 93.6/87.3 against 90.9/81.8, MNLI-m 89.0
against 86.6, SST-2 95.3 against 93.7. That's a substantial gap to open with 900,000 fewer optimizer
steps and the same 16GB of raw text, and it settles the question this rung was designed to answer
cleanly: the individual recipe corrections do compound, and BERT-large's original training was not
extracting anywhere close to what this recipe can pull out of the same corpus in a fraction of the
steps. That's an important result on its own, but it also reframes what "more training" should mean
going forward. If a corrected recipe this efficient is already ahead of the fully-trained original at
100K steps, the obvious next question is whether the remaining gap between "efficient but modest" and
"as good as this recipe can get" is a *data* ceiling, a *duration* ceiling, or both — and I specifically
want to test those as separate levers rather than pulling them together, for the same reason I insisted
on isolating batch size from total compute earlier in this study: if I add data and steps at once and
the numbers move, I won't know which lever did the work, and I'll have spent the much larger compute of
both changes to learn the answer to only one question.

Data is the cheaper lever to pull first, in the sense that it doesn't require deciding on a training
duration yet — I can hold steps fixed at exactly what rung 5 already used, 100,000, and change only how
much and how varied the pretraining text is. There's independent motivation to expect this matters.
Prior work outside this recipe has already shown that simply increasing pretraining data size, holding
everything else fixed, improves end-task performance — a general finding about pretraining, not
specific to any one architecture or objective. And several other groups building on masked or
autoregressive pretraining have separately moved to datasets substantially larger and more diverse than
the original 16GB BookCorpus+Wikipedia corpus, though not all of those datasets can be obtained or
released, which limits how directly their results can be compared to anything reproducible here. The
external permutation-based model referenced earlier in this study as a data point, for instance, is
trained on nearly ten times more data than original BERT and sees roughly four times as many total
sequences over the course of pretraining once its batch size and step count are accounted for — which
means any gap between that model and this recipe, measured so far only at BERT-large's original 16GB
scale, is currently conflating an architecture/objective difference with a scale difference. I can't
close that conflation without actually training on comparable data volume myself.

So the goal for this rung is to gather as much English-language pretraining text as practically
available, across a genuinely varied set of sources rather than one very large source, and add it to the
existing 16GB Books+Wikipedia corpus, while holding every other setting — architecture, masking, input
format, batch size, vocabulary, and step count — exactly as rung 5 left them. Three additional corpora
are worth combining, each chosen for a different reason. First, a news-crawl corpus built from the
English portion of a broad web crawl of news articles, spanning several years of publication — a
different register from Wikipedia's reference prose or BookCorpus's narrative fiction, contributing
current-events and journalistic language at a scale (tens of gigabytes after filtering) that's a
meaningful fraction of the total on its own. Second, an open-source recreation of a large web-text
corpus built from URLs shared on a social platform and filtered by community upvotes — a proxy for
curated general-interest web content, again with its own register distinct from the first two sources.
Third, a smaller corpus filtered specifically to match the story-like style of a set of common-sense
reasoning benchmarks — included less for its raw size and more because narrative, story-structured text
is a genre the other three sources underrepresent, and diversity of register is a variable I explicitly
want to vary alongside raw byte count, not conflate with it. Together with the original 16GB, this
totals a large multiple of the original corpus size — comparable in order of magnitude to what other
groups have used, though gathered independently since not all of those groups' original data is
obtainable.

I want to flag, honestly, that this rung conflates two things I can't fully separate with the sources
available: data *size* and data *diversity* both increase together, since the added corpora are both
larger in aggregate and drawn from different domains than the original two sources. A cleaner
experiment would hold total byte count fixed while varying only how many distinct sources contribute to
it, or vice versa — but that would require either discarding usable text to match a byte budget or
finding same-domain data at a much larger scale than what's available, and I don't think either
substitution is worth the data I'd have to throw away or the domains I'd have to sacrifice to get it. So
this rung will report on the combined effect of more and more varied data together, without being able
to attribute a downstream shift specifically to one or the other; that's a real limitation on how
finely I can interpret whatever this rung shows, one to keep in mind rather than a reason to hold off on
running it.

The prediction I want to state plainly, without presuming the size of the effect: given that the
general finding elsewhere is that more pretraining data improves downstream performance, and given that
rung 5 already showed this recipe efficiently extracts signal from a fixed 16GB corpus in far fewer
steps than the original needed, I expect performance to improve further across SQuAD, MNLI-m, and
SST-2 relative to rung 5's numbers, holding steps fixed at 100,000. Whether that improvement is small
(the recipe corrections already captured most of the available headroom, and data volume is a
second-order effect at this step count) or large (BERT-large was significantly data-starved on top of
being step-starved, and the recipe corrections and data volume are close to independent, additive
sources of gain) is exactly what this rung is designed to distinguish — and the answer will shape how
much weight the next rung, which holds data fixed and extends training duration instead, should expect
to be doing on its own.

So: combine the three additional English-language corpora described above with the original 16GB
BookCorpus+Wikipedia data for a substantially larger and more varied pretraining set; hold the combined
recipe (dynamic masking, full-sentences without NSP, 8K batch, byte-level BPE), BERT-large architecture,
and step count (100,000) exactly as rung 5 established them; and evaluate on the same fixed protocol —
SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy — against rung 5's own numbers.
