Before I optimize anything I want the most direct, obviously-correct expression of the task — a version
whose only job is to be right and to hand me a wall-clock number that everything afterward is measured
against. The task reads almost as a single sentence: for each station, compute the minimum, mean, and
maximum temperature over one billion `station;temperature` lines, then print the stations sorted
alphabetically and rounded to one decimal. The temptation to reach straight for raw bytes, memory mapping,
and eight threads is real — the context even lists all of those as available techniques — but a reference
implementation should be the task definition transcribed, not a clever thing. So I deliberately stay at the
level of `String`s, boxed doubles, and a single thread, and let its slowness become the budget the rest of
the ladder spends down.

Let me first make the scale concrete, because the whole design conversation from here on is about what one
billion of anything costs. The input is ~12 GB across 1,000,000,000 rows, which is about twelve bytes per
row on average: a station name (the primary dataset has 413 distinct stations, names running a handful of
bytes each), one `;`, a temperature that is three-to-five characters (`D.D`, `DD.D`, or a leading `-`), and
a newline. Twelve bytes times a billion is twelve gigabytes — the arithmetic checks against the stated file
size, so my mental model of the row shape is right. The important consequence is that the *arithmetic* the
task asks for is trivial: per row it is one comparison against a running min, one against a running max, one
addition into a sum, one increment of a count. Four integer or floating operations. If those four ops were
all I paid for, a billion of them at a couple of nanoseconds each would finish in a couple of seconds on a
single core. So anything slower than that is *overhead*, not computation, and naming exactly which overhead
I am choosing to accept is the real content of this rung.

I can put a second floor under that intuition from the I/O side. The 12 GB has to be read once; the machine
has 128 GB of RAM, so after the first pass the whole file lives in the page cache and subsequent touches are
memory-bandwidth-bound, not disk-bound. Even at a conservative single-core streaming rate of order 10 GB/s,
moving 12 GB of bytes through one core is on the order of a second. So both the arithmetic floor and the
raw-bytes floor sit around a second or two. If the idiomatic version lands two orders of magnitude above
that, essentially all of the gap is layers of abstraction I chose to accept, and each layer is a thing a
later rung can peel off. That framing — a small hard floor, a large soft overhead — is the lens I want for
the whole ladder.

Now the mapping from the task sentence onto idiomatic Java. Reading the file as lines is
`Files.lines(Paths.get(FILE))`, which hands me a `Stream<String>` whose elements are already-decoded lines.
Each line is `station;temperature`, so `l.split(";")` cuts it into the two fields, and I wrap the result in
a tiny `Measurement` record whose canonical constructor parses the second field with `Double.parseDouble`.
Now I have a stream of `(station, value)` pairs, and "for each station, aggregate" is exactly what
`Collectors.groupingBy` expresses: group by `m.station()`, and fold each group with a custom collector.

The shape of that per-station aggregate is the one design decision in this rung that I actually have to get
right, because it propagates into every rung after. I need to end with three numbers per station — min,
mean, max — but I cannot carry a running *mean* directly, because a mean is not composable: knowing the
mean of the first half of a station's readings and the mean of the second half does not let me recover the
mean of the whole unless I also know how many readings went into each. So the aggregate has to be four
numbers — running `min`, running `max`, running `sum`, and a `count` — from which the mean is `sum / count`
at the end. That quadruple `(min, max, sum, count)` is a *sufficient statistic* for `(min, mean, max)`: it
holds everything the answer needs and nothing it does not. So I define a small mutable
`MeasurementAggregator` over those four fields and write a `Collector.of(...)`. The supplier makes a fresh
aggregator with `min` seeded to `+∞` and `max` to `−∞` so the very first real value wins both comparisons.
The accumulator folds one measurement in: `min = Math.min(min, v)`, `max = Math.max(max, v)`, `sum += v`,
`count++`.

The combiner is the piece I want to write carefully even though this rung never runs it, because it is the
seam the entire ladder will later pull on. It merges two aggregators elementwise: `min` of the two mins,
`max` of the two maxes, the two sums added, the two counts added. What makes this legitimate is that each of
those four merges is over an *associative and commutative* operation — `min`, `max`, integer `+`, integer
`+` — so however I later cut the billion rows into pieces and however I later glue the pieces back together,
the tree of merges collapses to the same answer as a single left-to-right fold. Concretely, for any three
partial aggregates `a`, `b`, `c`, `merge(merge(a,b),c) = merge(a,merge(b,c))` because `min` and `+` each
satisfy that law componentwise. I am writing this rung single-threaded, but I am writing the *parallel-ready
shape* now: the moment I want to hand different slices of the file to different cores, this exact combiner is
what lets me sum their partial results at the end without a second thought. The finisher turns the
aggregator into a `ResultRow(min, sum/count, max)`.

Let me actually check the associativity claim on numbers rather than trust the algebra, since it is the one
property the whole ladder leans on. Take a station whose readings, split across three imagined workers, are
`a = (min 3, max 9, sum 12, count 2)`, `b = (min 1, max 5, sum 6, count 2)`, `c = (min 7, max 8, sum 15,
count 2)`. Merging left-to-right: `merge(a,b) = (min 1, max 9, sum 18, count 4)`, then `merge(that, c) =
(min 1, max 9, sum 33, count 6)`. Merging right-first: `merge(b,c) = (min 1, max 8, sum 21, count 4)`, then
`merge(a, that) = (min 1, max 9, sum 33, count 6)`. Identical quadruple, hence identical
`min/mean/max = 1 / 5.5 / 9`. The grouping order did not matter, which is exactly the guarantee I need
before I ever dare to cut the file into pieces and process them out of order.

Output needs the stations alphabetized, and a `TreeMap` keeps its keys sorted by definition, so I collect
into a `TreeMap<String, ResultRow>`. A `Map`'s default `toString` prints `{key=value, key=value, …}`, which
is exactly the required `{Station=…/…/…, …}` shape, so I lean on it directly rather than building the string
myself. The rounding rule — one fractional digit — I apply only at print time, in `ResultRow`'s `toString`,
as `Math.round(value * 10.0) / 10.0`.

I want to actually run the whole thing on a tiny input in my head, because "correct by construction" is a
claim I should be able to check, not just assert. Take three rows:

    Hamburg;12.0
    Hamburg;-5.3
    Bulawayo;8.9

Grouping by station gives two groups. Hamburg folds `12.0` then `-5.3`: `min = min(+∞, 12.0) = 12.0`, then
`min(12.0, -5.3) = -5.3`; `max = 12.0`; `sum = 12.0 + (-5.3) = 6.7`; `count = 2`; so `mean = 6.7 / 2 =
3.35`. Bulawayo folds a single `8.9`: `min = max = 8.9`, `mean = 8.9`. The `TreeMap` orders the keys, and
`'B' < 'H'`, so Bulawayo prints first. Now the rounding, which is the part most likely to be subtly wrong.
For Hamburg's mean, `Math.round(3.35 * 10.0) / 10.0 = Math.round(33.5) / 10.0`, and `Math.round` rounds
half up, so `34 / 10.0 = 3.4`. For its min, `Math.round(-5.3 * 10.0) / 10.0 = Math.round(-53.0) / 10.0 =
-5.3`. So the line reads `Hamburg=-5.3/3.4/12.0`, and the whole output is `{Bulawayo=8.9/8.9/8.9,
Hamburg=-5.3/3.4/12.0}`. That is the format the task demands, produced with no special-case code — which is
the payoff of leaning on `TreeMap.toString`. One honest note falls out of tracing it: the challenge
specifies IEEE `roundTowardPositive`, and `Math.round`'s round-half-up is the *naive* rounding, not that
exact rule; the faster entries will tighten it. For a readable reference this is the honest form, and it is
right on every value that is not exactly on a half-way boundary, which is the overwhelming majority.

One detail in the finisher is worth tracing separately, because it is the kind of thing that looks like a
bug and is not. The mean is not computed as a clean `sum / count`; it is `(Math.round(sum * 10.0) / 10.0) /
count`, which rounds the *sum* to one decimal before dividing, and then `ResultRow.toString` rounds the
result again. Following Hamburg through both roundings: `sum = 6.7`, so `Math.round(6.7 * 10.0) / 10.0 =
Math.round(67.0) / 10.0 = 6.7`, then `/ count = 6.7 / 2 = 3.35`, and finally `round(3.35) = 3.4`. It
double-rounds but still lands on `3.4`, the same value the clean `sum/count` would print. I note it as a
quirk of the naive rounding rather than a defect: on this data the two roundings agree, and tightening the
rule to the exact `roundTowardPositive` is exactly the kind of correctness polish a faster entry can take on
without changing the aggregate it rounds.

So the program is correct. It is also slow by construction, and naming precisely where the time goes is what
sets up the entire ladder, so let me account for it rather than wave at it. There are two independent
sources of cost, and I can size each.

The first is that it is single-threaded. `Files.lines(...).collect(...)` runs the whole pipeline on one
core while the other seven sit idle. The evaluation machine pins me to eight cores, so at best I am using an
eighth of the hardware; there is an up-to-8× ceiling sitting entirely untouched. That is not a subtle
constant factor — it is a whole order-of-magnitude-adjacent multiplier that the parallel-ready combiner I
just wrote is precisely designed to unlock later.

The second is allocation, and this is the one that makes the single core itself slow, independently of the
idle seven. Walk one row through the pipeline and count the objects born: `Files.lines` hands me a freshly
decoded `String` for the line; `split(";")` allocates a `String[]` array *and* two new substring `String`s
for the fields; and the `Measurement` record is a fourth object. That is on the order of four short-lived
objects per row, so across a billion rows I am asking the allocator to produce roughly four billion objects
whose entire lifetime is a single loop iteration. Every one of them floods the young generation, and the
garbage collector will be running essentially the whole job, copying survivors and reclaiming the dead — a
cost that has nothing to do with min/max/sum and everything to do with the level of abstraction I chose.

It is worth turning that object count into bytes, because the size is the part that surprises. A decoded
line `String` for a ~12-byte line is, with compact strings, its object header plus the Latin-1 bytes —
call it ~30 bytes. The `String[]` from `split` is a small array object plus two references, another ~32.
The two substring `String`s are two more headers plus their bytes, ~40 together. The `Measurement` record
holds a reference and a `double`, ~32. That is very roughly 130–140 bytes of allocation per row, and across
a billion rows it is on the order of 140 GB of garbage churned through the young generation to process a
12 GB file — more than ten times the input size, all of it dead within one loop iteration. With a young
generation sized in the hundreds of megabytes to a couple of gigabytes, 140 GB of allocation means on the
order of a hundred minor collections over the run, each pausing the one working thread to scan for the tiny
handful of survivors (the aggregators and the map). None of that copying moves the answer forward; it is
pure tax on having expressed each row as objects. On
top of the allocation, the *work* inside each of those layers is heavier than the data warrants:
`split(";")` runs a regex-style scan per line to find a single fixed byte; `Double.parseDouble` is a
general-purpose floating-point parser handling exponents, infinities, and arbitrary precision when the data
is only ever `D.D`/`DD.D` with an optional minus; and `groupingBy` hashes a full `String` key and walks a
`HashMap` bucket for every one of the billion rows, when there are only a few hundred distinct keys.

That last one hides a redundancy worth spelling out, because it points at a specific later lever. Think about
how many times the *bytes of the station name* get touched per row. The UTF-8 decoder walks them once to
build the line `String`. The `split(";")` walks them again to find the delimiter and copy out the substring.
`groupingBy` then calls `String.hashCode`, which walks every character a third time computing `31*h + c`,
and on a bucket hit `equals` walks them a fourth time to confirm the match. So a seven-byte name is scanned
four-plus times per row, a billion rows over, when the delimiter search and the hash could in principle be
folded into a *single* pass — find the `;` and accumulate the hash in the same walk, and never build a
`String` at all. I am not doing that here; I am noting that the idiomatic version pays for the name bytes
several times over, and that collapsing those passes into one is a concrete constant-factor waiting to be
collected.

There is a third cost hiding in the very first step. Reading the file as *decoded lines* forces the JVM to
UTF-8-decode all ~12 GB into `char`/`String` data before I have even looked at a byte. But every byte I
actually care about — `;`, the digits, `-`, `.`, `\n` — is plain ASCII. I am paying a full 12 GB decode
pass to recover information that was already sitting in the raw bytes. That decode is pure tax against the
particular structure of this input.

None of this is *wrong*. It is the honest cost of expressing the problem at the level of `String`s and boxed
doubles on a single thread, and each layer is exactly a lever the ladder will later pull: get off decoded
lines and onto raw bytes; get off `String` keys and onto byte ranges; get off `parseDouble` and onto an
integer trick that exploits the fixed `D.D` shape; get off one core and onto eight via the associative
combiner; and eventually get off the JVM's startup and warmup entirely. I am not going to spend any of those
levers now — spending them now would rob me of the clean attribution I get by changing one layer at a time
and reading each change off the wall-clock.

Before I commit, I should be honest that even within "idiomatic and single-threaded" there were choices, and
that I picked the clear one rather than the least-slow one on purpose. I could have manually looped a
`BufferedReader.readLine()` instead of the stream, but that removes only the collector machinery while
keeping the `String`, the `split`, and the `parseDouble` — a marginal change that would muddy the reference
without teaching me anything, so I keep the stream, which reads most like the task sentence. I could have
called `Files.readAllLines`, but that materializes all billion lines in a `List` at once — on the order of
140 GB of live `String`s — and simply runs the heap out of memory; it is not merely slower, it does not
finish, so it is disqualified outright. And I could have flipped `Files.lines(...)` to `parallelStream()`
to grab the idle cores for free, but the line-oriented spliterator behind `Files.lines` does not split
cleanly — it cannot cheaply find where to cut a stream of variable-length lines — so a naive `parallel()`
tends not to scale and, worse, would smuggle a real optimization into what is supposed to be the untouched
reference. Proper parallelism means choosing *where* to cut the file myself, which is a deliberate next-rung
move built on the associative combiner I just wrote, not a flag I toggle here. So the design-space walk ends
where it started: the honest reference is the sequential stream.

There is one representation choice worth flagging honestly, because it is both a correctness subtlety and a
foreshadowing of a later lever. I accumulate `sum` as a `double`. Over a billion additions of values up to
99.9, the running sum climbs toward ~10^11, and a `double` carries about 15–16 significant digits, so once
the accumulator is large the low-order bits of each freshly added tenth start falling off the end of the
mantissa — floating-point addition is not associative, and a billion of them drift. On this task the mean is
rounded to a single decimal before printing, so the drift almost never crosses a rounding boundary and the
output stays correct; but "almost never" is not "never," and the clean fix is to carry the sum as an exact
scaled integer instead of a `double`. I leave it as a `double` here because the reference should read like
the task, and I note the exact-integer representation as a correctness *and* speed improvement a later rung
can adopt — the data is `D.D`, one decimal, so multiplying by ten makes every value an exact integer with no
float in sight.

One cost I explicitly do *not* count against this rung is JVM startup and JIT warmup. The VM takes a couple
of hundred milliseconds to start and the hot loop runs interpreted for its first iterations before the JIT
compiles it — but against a run measured in minutes, a few hundred milliseconds is well under a percent, so
it rounds to nothing here. I flag it only because that same fixed cost stops being negligible the instant a
later rung drives the whole job down into a couple of seconds; what is invisible now becomes a real fraction
then. It is a cost that grows in *relative* importance precisely as the ladder succeeds.

Setting the bar for what this rung should measure: the metric is end-to-end wall-clock in `m:ss.mmm`, and
this is the reference number the challenge publishes for the idiomatic implementation, in the region of four
to five minutes. My own accounting says why: a couple of seconds of genuine arithmetic, buried under a
single-core cap and a per-row overhead — allocation, GC, a 12 GB decode, a regex split, a general float
parse, and a `String`-hashing map — that is spending far more than the arithmetic itself. The falsifiable
prediction I am carrying into the next rung is specific: because the time is dominated by idle cores and
per-row overhead rather than by the aggregation, the two structural moves that should each buy a large
multiplier are (1) using all eight cores through the associative combiner and (2) collapsing the per-row
allocation and decode by working on raw bytes — and those two are independent, so they should compound
rather than merely add. For now this is the baseline: idiomatic, correct, and the honest reference every
faster rung is measured against. The full module is in the answer.
