## Research question

Read one billion `station;temperature` lines and emit, for each distinct station, its min/mean/max sorted alphabetically as `<min>/<mean>/<max>` rounded to one decimal with IEEE-754 `roundTowardPositive`. Stations are UTF-8, 1–100 bytes, with no `;` or `\n`. Temperatures are in [−99.9, 99.9] with exactly one fractional digit. Input is ~12 GB; there are at most 10,000 distinct stations. The entry is a single Java source file with no external dependencies, and the answer must be computed at runtime.

Evaluation pins each entry to **eight cores** of a Hetzner AX161 (32-core AMD EPYC 7502P, Zen2, 2.5 GHz, 128 GB RAM), SMT off, Turbo off, ext4. The input, output format, rounding rule, and hardware are fixed; the ranking is wall-clock time from process launch to the last output byte. Lower is better.

The task is almost entirely **memory- and parse-bound**: a billion delimiter searches, parses, hash-map probes, and ~12 GB of bytes moving through the cores. Arithmetic per row is trivial; the cost lies in I/O, branch mispredictions, cache misses, allocation, UTF-8 handling, and JVM startup/JIT warmup.

## Prior art / Background / Baselines

Current entries apply these techniques:

- **Java streams + collections.** Express the pipeline idiomatically with `Files.lines()`, `String.split(";")`, `Double.parseDouble`, and `Collectors.groupingBy` into a `TreeMap`. The reference implementation finishes in **04:49.679** (m:ss.mmm); every entry must beat this number.

- **Memory-mapped I/O.** Map the file with `FileChannel.map` or `MemorySegment` so threads read directly from the page cache without `read()` syscalls or intermediate buffers.

- **Multithreading.** Split the file into segments, run one per core, then merge partial results.

- **Vector API.** Use `ByteVector` to load 16–32 bytes and compare against `';'` in one SIMD instruction, producing a bitmask of delimiter positions. Requires `--add-modules jdk.incubator.vector` / `--enable-preview`.

- **SWAR.** Find delimiter bytes in a 64-bit `long` with the bit-twiddling identity `(w - 0x0101…01) & ~w & 0x8080…80` and `Long.numberOfTrailingZeros`.

- **Raw off-heap access.** Read mapped memory directly with `sun.misc.Unsafe.getLong` or `java.lang.foreign.MemorySegment`, avoiding bounds checks and object headers.

- **GraalVM `native-image`.** Compile the program ahead of time to a standalone binary, eliminating JVM startup and JIT warmup.

## Fixed substrate / Code framework

Every entry implements a `main` that reads `./measurements.txt`, aggregates min/mean/max per station, and prints `{Station=min/mean/max, …}` sorted by station. The mean is `sum/count`; only the output is rounded. Fast entries store temperatures as scaled integers (×10, so [−999, 999] fits in `short` or `int`) and perform min/max/sum/count as integer operations.

The pipeline is fixed: read bytes → split key from value → parse value → look up and update the station aggregate → merge partial aggregates → sort and print. A minimal runnable scaffold that satisfies the contract is:

```java
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class CalculateAverage {
    static class Result {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        long sum;
        int count;
        void add(int v) {
            min = Math.min(min, v);
            max = Math.max(max, v);
            sum += v;
            count++;
        }
    }

    public static void main(String[] args) throws Exception {
        Map<String, Result> map = new TreeMap<>();
        try (var br = Files.newBufferedReader(Path.of("./measurements.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                int sep = line.indexOf(';');
                String station = line.substring(0, sep);
                int v = (int) Math.round(Double.parseDouble(line.substring(sep + 1)) * 10);
                map.computeIfAbsent(station, k -> new Result()).add(v);
            }
        }
        StringBuilder out = new StringBuilder("{");
        for (var e : map.entrySet()) {
            Result r = e.getValue();
            if (out.length() > 1) out.append(", ");
            double mean = Math.ceil((double) r.sum / r.count) / 10.0;
            out.append(e.getKey()).append('=')
               .append(r.min / 10.0).append('/')
               .append(mean).append('/')
               .append(r.max / 10.0);
        }
        System.out.println(out.append('}'));
    }
}
```

This scaffold is correct and runnable.

## Editable interface

The editable part is the implementation of the five pipeline stages: I/O and byte supply, delimiter search, temperature parse, per-station aggregate lookup/update, and final merge/sort/output. The evaluation harness fixes the input path, the output format, the JDK/JVM flags, and the core count. A valid entry may use any language feature or API available under those flags, including memory mapping, threads, SIMD intrinsics, off-heap access, or ahead-of-time compilation.

## Evaluation settings

The ranking metric is wall-clock time (m:ss.mmm) from process launch to last output byte on the fixed machine: eight cores of the AMD EPYC 7502P, 128 GB RAM, SMT and Turbo off, ext4. The input is 1,000,000,000 rows generated by the challenge's own generator (413 distinct stations in the primary dataset; a separate 10,000-station run stresses the key-set ceiling). Correctness is checked against the reference output before timing. Reported numbers are the published leaderboard results for each entry. Lower is better.
