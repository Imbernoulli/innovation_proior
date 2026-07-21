Before I optimize anything I want the most direct, obviously-correct expression of the task — a version
whose only job is to be right and to hand me a wall-clock number everything afterward is measured against.
For each station, compute min, mean, and max over one billion `station;temperature` lines, then print the
stations sorted alphabetically and rounded to one decimal. The context lists raw bytes, memory mapping, and
eight threads as available techniques, but a reference implementation should be the task definition
transcribed, not a clever thing. So I stay at the level of `String`s, boxed doubles, and a single thread,
and let its slowness become the budget the rest of the ladder spends down.

First make the scale concrete, because the whole design conversation from here is about what one billion of
anything costs. The input is ~12 GB across 1,000,000,000 rows, about twelve bytes per row: a station name
(413 distinct stations, a handful of bytes each), one `;`, a temperature of three-to-five characters, a
newline. Twelve bytes times a billion is twelve gigabytes, which checks against the stated file size. The
consequence is that the *arithmetic* the task asks for is trivial: per row, one min compare, one max
compare, one add, one increment. If those four ops were all I paid for, a billion of them at a couple of
nanoseconds each would finish in a couple of seconds on one core. So anything slower is *overhead*, not
computation, and naming which overhead I choose to accept is the real content of this rung. The I/O side
puts a second floor under that: the 12 GB is read once, and with 128 GB of RAM the whole file lives in page
cache after the first pass, so subsequent touches are memory-bandwidth-bound — order a second to move 12 GB
through one core. Both floors sit around a second or two; if the idiomatic version lands two orders of
magnitude above that, essentially all of the gap is layers of abstraction I chose, and each layer is a thing
a later rung can peel off.

Now the mapping onto idiomatic Java. `Files.lines(Paths.get(FILE))` hands me a `Stream<String>` of
already-decoded lines; `l.split(";")` cuts each into two fields wrapped in a tiny `Measurement` record whose
constructor parses the second field with `Double.parseDouble`; and "for each station, aggregate" is exactly
`Collectors.groupingBy`.

The shape of the per-station aggregate is the one decision here I have to get right, because it propagates
into every rung after. I cannot carry a running *mean* directly — a mean is not composable: the means of two
halves do not recover the whole unless I also know the two counts. So the aggregate is four numbers — `min`,
`max`, `sum`, `count` — from which the mean is `sum/count` at the end. That quadruple is the minimal
*sufficient statistic* for `(min, mean, max)`. I define a mutable `MeasurementAggregator` over those fields
and a `Collector.of(...)`: the supplier seeds `min` to `+∞` and `max` to `−∞` so the first real value wins
both; the accumulator folds one measurement in.

The combiner is the piece I write carefully even though this rung never runs it, because it is the seam the
whole ladder will pull on. It merges two aggregators elementwise — min of mins, max of maxes, sums added,
counts added — and each of those operations is associative and commutative, so however I later cut the
billion rows into pieces and glue them back, the tree of merges collapses to the same answer as a single
left-to-right fold. I am writing single-threaded but in the *parallel-ready shape*: the moment I want to
hand slices to different cores, this exact combiner sums their partial results with no second thought. The
finisher turns the aggregator into a `ResultRow(min, sum/count, max)`.

Output needs stations alphabetized, and a `TreeMap` keeps its keys sorted by definition, so I collect into a
`TreeMap<String, ResultRow>` and lean on `Map.toString`'s `{key=value, …}` shape directly rather than
building the string myself. The rounding rule — one fractional digit — I apply only at print time as
`Math.round(value * 10.0) / 10.0`. One honest caveat: the challenge specifies IEEE `roundTowardPositive`,
and `Math.round`'s round-half-up is the *naive* rule, not that exact one; it is right on every value not
exactly on a half-way boundary, which is the overwhelming majority, and tightening it is correctness polish a
faster entry can take. The finisher also double-rounds the mean — `(Math.round(sum*10.0)/10.0)/count`, then
`ResultRow.toString` rounds again — but on this data the two roundings agree, so I leave the readable form.

So the program is correct. It is also slow by construction, and naming precisely where the time goes is what
sets up the ladder. Two independent sources of cost.

The first is that it is single-threaded — the pipeline runs on one core while seven sit idle. The machine
pins me to eight cores, so there is an up-to-8× ceiling sitting entirely untouched, exactly what the
associative combiner is designed to unlock later.

The second is allocation, and this is what makes the single core itself slow. Walk one row through the
pipeline: `Files.lines` hands me a decoded `String`; `split(";")` allocates a `String[]` *and* two substring
`String`s; the `Measurement` record is a fourth object. Four short-lived objects per row, roughly 130–140
bytes of allocation, so across a billion rows on the order of 140 GB of garbage churned through the young
generation to process a 12 GB file — more than ten times the input size, all of it dead within one loop
iteration. That floods the young generation and keeps the garbage collector running essentially the whole
job, copying the tiny handful of survivors (the aggregators and the map); none of that copying moves the
answer forward. On top of the allocation, the *work* inside each layer is heavier than the data warrants:
`split(";")` runs a regex-style scan to find a single fixed byte; `Double.parseDouble` is a general float
parser handling exponents and arbitrary precision when the data is only ever `D.D`/`DD.D`; and `groupingBy`
hashes a full `String` key and walks a `HashMap` bucket for every row when there are only a few hundred
distinct keys.

That last one hides a redundancy pointing at a specific later lever. Count how many times the *bytes of the
station name* get touched per row: the UTF-8 decoder walks them once building the line, `split` walks them
again to find the delimiter and copy the substring, `String.hashCode` walks them a third time, and on a
bucket hit `equals` walks them a fourth. A seven-byte name scanned four-plus times per row, a billion rows
over — when the delimiter search and the hash could be folded into a single pass and never build a `String`
at all. And there is a third cost in the very first step: reading the file as *decoded lines* forces a full
12 GB UTF-8 decode before I have looked at a byte, when every byte I care about — `;`, the digits, `-`, `.`,
`\n` — is plain ASCII.

None of this is *wrong*. It is the honest cost of expressing the problem at the level of `String`s and boxed
doubles on one thread, and each layer is exactly a lever the ladder will pull: off decoded lines onto raw
bytes, off `String` keys onto byte ranges, off `parseDouble` onto an integer trick exploiting the fixed
`D.D` shape, off one core onto eight via the combiner, and eventually off the JVM's startup and warmup. I
spend none of them now — spending them now would rob me of the clean attribution I get by changing one layer
at a time and reading each change off the wall-clock.

Two choices within "idiomatic and single-threaded" I make deliberately. `Files.readAllLines` would
materialize all billion lines in a `List` — ~140 GB of live `String`s — and run the heap out of memory: not
merely slower, it does not finish. And flipping to `parallelStream()` to grab the idle cores would smuggle a
real optimization into the untouched reference, and worse the line-oriented spliterator behind `Files.lines`
cannot cheaply find where to cut a stream of variable-length lines, so it tends not to scale anyway. Proper
parallelism means choosing *where* to cut the file myself — a deliberate next-rung move built on the
combiner — not a flag I toggle here.

One representation choice worth flagging, both a correctness subtlety and a later lever: I accumulate `sum`
as a `double`. Over a billion additions the running sum climbs toward ~10^11, and a `double` carries ~15–16
significant digits, so once the accumulator is large the low bits of each added tenth fall off the mantissa —
a billion non-associative float adds drift. The mean is rounded to one decimal before printing, so the drift
almost never crosses a rounding boundary; but "almost never" is not "never," and the clean fix is an exact
scaled integer (the data is one decimal, so ×10 makes every value an exact integer). I leave the `double`
here and note the integer representation as a later correctness *and* speed win.

I do *not* count JVM startup and JIT warmup against this rung: a couple hundred milliseconds against a run
measured in minutes rounds to nothing. I flag it only because that same fixed cost stops being negligible the
instant a later rung drives the job into a couple of seconds — a cost that grows in *relative* importance
precisely as the ladder succeeds.

The metric is end-to-end wall-clock, and this is the reference the challenge publishes for the idiomatic
implementation, in the region of four to five minutes: a couple of seconds of genuine arithmetic buried under
a single-core cap and per-row overhead — allocation, GC, a 12 GB decode, a regex split, a general float
parse, a `String`-hashing map. Because the time is dominated by idle cores and per-row overhead rather than
by aggregation, the two structural moves that should each buy a large multiplier — all eight cores through
the combiner, and collapsing per-row allocation and decode by working on raw bytes — attack independent
costs, so they should compound rather than merely add. For now this is the baseline: idiomatic, correct, the
honest reference every faster rung is measured against. The full module is in the answer.
