**Problem.** Compute per-station min/mean/max over one billion `station;temperature` rows and print them
sorted and rounded to one decimal — correctly and as the reference everything else is measured against,
with no premature optimization.

**Key idea.** Transcribe the task definition directly into the Java streams idiom: read decoded lines with
`Files.lines`, split each into `station` and `value`, group by station with `Collectors.groupingBy`, and
fold each group with a custom four-field collector (min / max / sum / count) whose finisher emits
`min / (sum/count) / max`. Collect into a `TreeMap` so output is alphabetically sorted for free, and round
only at print time.

**Why it works.** Min/max/sum/count is the minimal sufficient statistic for min/mean/max, and the
combiner merges two partial aggregates elementwise — associative, so the same collector parallelizes later
without change. It is correct by construction because it is the problem statement expressed in stream
operations. It is also slow by construction: single-threaded, one `String` + `String[]` + boxed `Double`
+ record allocated per row, full UTF-8 decoding of all ~12 GB, a regex-style `split`, a general-purpose
`Double.parseDouble`, and `String`-keyed hashing. That cost is the baseline budget the ladder spends down.

**Change / code.** The reference single-threaded streams implementation.

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
