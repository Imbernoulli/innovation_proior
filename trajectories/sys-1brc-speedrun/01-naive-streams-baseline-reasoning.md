I want the most direct, obviously-correct expression of the task first, before I optimize anything — a
version whose only job is to be right and to give me a wall-clock number to beat. The task is: for each
station, the min, mean, and max temperature over a billion lines, printed sorted and rounded. So I reach
straight for the idiomatic Java that maps onto that sentence almost word for word.

Reading the file as lines is `Files.lines(Paths.get(FILE))` — that hands me a `Stream<String>` where each
element is one decoded line. Each line is `station;temperature`, so `l.split(";")` cuts it into the two
fields, and I wrap that in a small `Measurement` record that parses the second field with
`Double.parseDouble`. Now I have a stream of `(station, value)` pairs, and "for each station, aggregate"
is exactly what `Collectors.groupingBy` is for: group by `m.station()`, and for each group fold the values
with a custom collector.

The aggregate I need per station is four numbers — running min, running max, running sum, and a count —
because the mean is `sum/count` and I can't compute a streaming mean without the count. So I define a tiny
mutable `MeasurementAggregator` holding `min`, `max`, `sum`, `count`, and write a `Collector.of(...)` over
it: the supplier makes a fresh aggregator (min seeded to `+∞`, max to `−∞`), the accumulator folds one
measurement in (`min = Math.min(min, v)`, `max = Math.max(max, v)`, `sum += v`, `count++`), and the
combiner merges two aggregators elementwise — which I'll need the moment I ever parallelize, since
min/max/sum/count all combine associatively. The finisher turns the aggregator into a `ResultRow(min,
mean, max)` where `mean = sum/count`.

For output I need the stations in alphabetical order, and a `TreeMap` keeps its keys sorted by definition,
so I collect the grouped result into a `TreeMap<String, ResultRow>` and print it. The default `toString`
of a map prints `{key=value, key=value, …}`, which is exactly the required `{Station=…/…/…, …}` shape, so
I lean on it. The rounding rule — one fractional digit — I apply only at print time, in `ResultRow`'s
`toString`, as `Math.round(value * 10.0) / 10.0`. (The challenge specifies IEEE roundTowardPositive; the
baseline's `Math.round` is the naive version, and the faster entries will tighten it, but for a reference
implementation this is the honest, readable form.)

That is the whole program, and it is correct by construction — it is just the task definition transcribed
into stream operations. But I can already see where every second is going to go, and naming it now sets up
everything that follows. It is single-threaded: `Files.lines(...).collect(...)` runs on one core while the
other seven sit idle, so at best I'm using an eighth of the machine. It allocates relentlessly: every one
of the billion lines becomes a freshly decoded `String`, then a `String[]` from `split`, then a boxed
`Double` from `parseDouble`, then a `Measurement` object — that is on the order of four billion short-lived
objects flooding the young generation, and the garbage collector will be doing real work the whole run.
The `split(";")` compiles and applies a regch-style scan per line; `Double.parseDouble` is a
general-purpose float parser doing far more than the trivial `D.D`/`DD.D` shapes the data actually
contains; and `groupingBy` hashes a full `String` key and walks a `HashMap` per row. Reading the file as
*decoded lines* also means the JVM is UTF-8-decoding all ~12 GB into `char`/`String` data even though the
delimiter and digits I care about are plain ASCII bytes.

None of that is wrong — it is just the cost of expressing the problem at the level of `String`s and boxed
doubles and a single thread. The wall-clock this lands at is the budget the rest of the ladder spends:
every structural change downstream is going to be about *removing* one of these layers — getting off the
stream and onto raw bytes, off `String` keys and onto byte ranges, off `parseDouble` and onto integer
tricks, off one core and onto eight, off the JVM's startup onto an ahead-of-time binary. For now this is
the baseline: idiomatic, correct, and the reference everything else is measured against. The full module is
in the answer.
