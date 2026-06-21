Before optimizing anything, I want the most direct, obviously-correct expression of the task — a version whose only job is to be right and to hand me a wall-clock number that everything afterward is measured against. The task reads almost as a sentence: for each station, compute the minimum, mean, and maximum temperature over a billion `station;temperature` lines, then print the stations sorted alphabetically and rounded to one decimal. The temptation to reach for raw bytes and threads immediately is real, but a reference implementation should be the task definition transcribed, not a clever thing — so I deliberately stay at the level of `String`s, boxed `double`s, and a single thread, and let its slowness become the budget the rest of the ladder spends down.

The method is the **naive streams baseline**, idiomatic Java mapped onto the task almost word for word. Reading the file as lines is `Files.lines(Paths.get(FILE))`, which hands me a `Stream<String>` where each element is one already-decoded line. Each line is `station;temperature`, so `l.split(";")` cuts it into the two fields, and I wrap the result in a tiny `Measurement` record whose canonical constructor parses the second field with `Double.parseDouble`. Now I have a stream of `(station, value)` pairs, and "for each station, aggregate" is exactly what `Collectors.groupingBy` expresses: group by `m.station()`, and fold each group with a custom collector.

The per-station aggregate has to be four numbers — running `min`, running `max`, running `sum`, and a `count` — because the mean is $\text{sum}/\text{count}$ and there is no way to keep a streaming mean without carrying the count alongside the sum. So I define a small mutable `MeasurementAggregator` holding those four fields and write a `Collector.of(...)` over it. The supplier makes a fresh aggregator with `min` seeded to $+\infty$ and `max` to $-\infty$ so the first real value always wins both comparisons. The accumulator folds one measurement in: `min = Math.min(min, v)`, `max = Math.max(max, v)`, `sum += v`, `count++`. The combiner merges two aggregators elementwise — `min` of the mins, `max` of the maxes, summed sums, summed counts. That combiner is the load-bearing piece: $(\min,\max,\text{sum},\text{count})$ is a sufficient statistic for $(\min,\text{mean},\max)$, and merging it elementwise is *associative*, so the very same collector parallelizes later with no change to its logic — I am writing the parallel-ready shape now even though this rung runs on one thread. The finisher turns the aggregator into a `ResultRow(min, sum/count, max)`.

Output needs the stations in alphabetical order, and a `TreeMap` keeps its keys sorted by definition, so I collect the grouped result into a `TreeMap<String, ResultRow>`. A `Map`'s default `toString` prints `{key=value, key=value, …}`, which is exactly the required `{Station=…/…/…, …}` shape, so I lean on it directly. The rounding rule — one fractional digit — I apply only at print time, in `ResultRow`'s `toString`, as `Math.round(value * 10.0) / 10.0`. The challenge specifies IEEE `roundTowardPositive`; this `Math.round` form is the naive rounding and the faster entries will tighten it, but for a readable reference it is the honest form.

That is the whole program, and it is correct by construction because it is the problem statement expressed in stream operations. It is also slow by construction, and naming where the time goes sets up the entire ladder. It is single-threaded — `Files.lines(...).collect(...)` runs on one core while the other seven idle, so at best an eighth of the machine is in use. It allocates relentlessly: every one of the billion lines becomes a freshly decoded `String`, then a `String[]` from `split`, then a boxed `Double` from `parseDouble`, then a `Measurement` — on the order of four billion short-lived objects flooding the young generation, with the garbage collector working the whole run. `split(";")` runs a regex-style scan per line; `Double.parseDouble` is a general float parser doing far more than the trivial `D.D`/`DD.D` shapes the data actually contains; and `groupingBy` hashes a full `String` key into a `HashMap` per row. Reading the file as *decoded lines* also forces the JVM to UTF-8-decode all ~12 GB into `char` data, even though the delimiter and digits I care about are plain ASCII bytes. None of this is wrong — it is simply the cost of expressing the problem at the level of `String`s and boxed doubles on a single thread, and every structural change downstream is about removing one of these layers.

```java
public class CalculateAverage_baseline {

    private static final String FILE = "./measurements.txt";

    private static record Measurement(String station, double value) {
        private Measurement(String[] parts) {
            this(parts[0], Double.parseDouble(parts[1]));
        }
    }

    private static record ResultRow(double min, double mean, double max) {
        public String toString() {
            return round(min) + "/" + round(mean) + "/" + round(max);
        }
        private double round(double value) {
            return Math.round(value * 10.0) / 10.0;
        }
    };

    private static class MeasurementAggregator {
        private double min = Double.POSITIVE_INFINITY;
        private double max = Double.NEGATIVE_INFINITY;
        private double sum;
        private long count;
    }

    public static void main(String[] args) throws IOException {
        Collector<Measurement, MeasurementAggregator, ResultRow> collector = Collector.of(
                MeasurementAggregator::new,
                (a, m) -> {
                    a.min = Math.min(a.min, m.value);
                    a.max = Math.max(a.max, m.value);
                    a.sum += m.value;
                    a.count++;
                },
                (agg1, agg2) -> {
                    var res = new MeasurementAggregator();
                    res.min = Math.min(agg1.min, agg2.min);
                    res.max = Math.max(agg1.max, agg2.max);
                    res.sum = agg1.sum + agg2.sum;
                    res.count = agg1.count + agg2.count;
                    return res;
                },
                agg -> new ResultRow(agg.min, (Math.round(agg.sum * 10.0) / 10.0) / agg.count, agg.max));

        Map<String, ResultRow> measurements = new TreeMap<>(Files.lines(Paths.get(FILE))
                .map(l -> new Measurement(l.split(";")))
                .collect(groupingBy(m -> m.station(), collector)));

        System.out.println(measurements);
    }
}
```
