## Research question

The task is deliberately tiny to state and brutal to optimize: read a text file of one billion lines,
each `station;temperature` (a UTF-8 station name of 1–100 bytes containing neither `;` nor `\n`, then a
temperature in [−99.9, 99.9] with exactly one fractional digit), and for every distinct station compute
its **minimum, mean, and maximum**. Emit the stations sorted alphabetically, each as `<min>/<mean>/<max>`
rounded to one decimal place with IEEE-754 `roundTowardPositive` semantics. The file is ~12 GB on disk;
there are at most 10,000 distinct station names (the reference dataset has 413). It must run as a single
Java source file with no external dependencies, and the result must be computed at runtime — no baking the
answer into a native image at build time.

Everything that defines the *problem* is frozen: the billion rows, the output format, the rounding rule,
the 10,000-station ceiling, and the hardware. The hardware is a Hetzner AX161 — a 32-core AMD EPYC 7502P
(Zen2) at 2.5 GHz with 128 GB RAM — but the evaluation pins each entry to **eight cores**, with SMT
disabled and Turbo Boost off, on an ext4 filesystem. Because the answer is fixed and the machine is fixed,
the only free variable is *how fast the program runs end to end* — wall-clock from launch to the last byte
of output. That is the single number the ladder is ranked on, and lower is better.

What makes the problem interesting is that it is almost entirely **memory- and parse-bound**, not
compute-bound. A billion lines is a billion delimiter searches, a billion number parses, a billion
hash-map probes, and ~12 GB of bytes that have to move from the page cache through the cores. The
arithmetic per row — three comparisons and an add — is trivial; the cost is everywhere *around* it: I/O,
branch mispredictions, cache misses, allocation, UTF-8 decoding, and the JVM's own startup and JIT
warmup. Every rung below is one structural attack on that overhead.

## Prior art before the first rung

There is no published literature to climb out of here — the lineage is the leaderboard itself, a public
record where each entry is a single self-contained Java program and the only currency is wall-clock
seconds. The relevant background is the toolbox modern Java hands a systems programmer, and which parts
of it the early entries had not yet reached for:

- **The Java streams + collections idiom.** `Files.lines()` returns a `Stream<String>` of decoded lines;
  `String.split(";")` cuts each into fields; `Collectors.groupingBy` plus a custom `Collector` folds them
  into per-key aggregates; a `TreeMap` gives the sorted output for free. This is the natural, idiomatic
  way to express the task, and it is where the baseline lives.
- **Memory-mapped I/O.** `FileChannel.map` (and, in newer JDKs, `MemorySegment` over an `Arena`) maps the
  file into the address space so the program reads bytes directly from the page cache without
  `read()` syscalls or intermediate buffers. The file can be sliced into segments and a thread given each.
- **`java.util.concurrent` and bare `Thread`s.** Eight cores sit idle unless the work is split across them;
  `Stream.parallel()`, an `ExecutorService`, or hand-managed `Thread[]` all spread segments over cores,
  with a final merge step combining the per-thread partial results.
- **The incubating Vector API (`jdk.incubator.vector`).** `ByteVector` exposes the CPU's SIMD lanes
  portably: load 16 or 32 bytes at once, compare them all against `';'` in one instruction, and read off
  a bitmask of matches. It needs `--add-modules jdk.incubator.vector` / `--enable-preview`.
- **SWAR — "SIMD within a register".** The same data-parallel idea using only plain 64-bit `long`
  arithmetic: load 8 bytes as one `long` and find a target byte with the classic bit-twiddling identity
  `(w - 0x0101…01) & ~w & 0x8080…80`, which lights up the high bit of any zero byte. No special
  instructions, no incubator module — just multiply, subtract, and `Long.numberOfTrailingZeros`.
- **`sun.misc.Unsafe` and `java.lang.foreign`.** Raw off-heap reads: given a memory address, `getLong`
  fetches 8 bytes with no bounds check and no object header. `MemorySegment` is the supported successor;
  `Unsafe` is the unsupported but maximally bare path the fastest entries still use.
- **GraalVM `native-image`.** Ahead-of-time compilation to a standalone native binary, eliminating JVM
  startup and JIT warmup — which, on a job that finishes in seconds, is a meaningful slice of wall-clock.

The baseline the ladder starts from is the challenge's own reference implementation: `Files.lines()` over
the file, `String.split(";")` per line, `Double.parseDouble` on the value, `groupingBy(station, …)` with a
custom min/max/sum/count collector, collected into a `TreeMap` and printed. Single-threaded, fully
idiomatic, allocating a `String` and a `String[]` and a boxed `Double` per row. On the evaluation machine
it finishes in **04:49.679** (m:ss.mmm). That is the number every rung below has to beat.

## The fixed substrate

The contract every entry fills is the same: a `main` that reads `./measurements.txt`, aggregates min/mean/
max per station, and prints `{Station=min/mean/max, …}` sorted by station, with each value rounded to one
decimal. The mean is computed as `sum/count` and only the *output* is rounded; internally the fast entries
keep temperatures as scaled integers (one fractional digit means the value times ten is an integer in
[−999, 999], so a `short` or `int` holds it exactly and min/max/sum become pure integer ops). The merge of
two partial aggregates is associative — `min`/`max`/`sum`/`count` combine elementwise — which is what makes
the whole thing parallelizable: split the file, aggregate each piece independently, combine at the end.

The frozen scaffold, then, is "read bytes → split key from value → parse value → look up the key's
aggregate and update it → merge partials → sort and print." Each rung is one structural change to *how*
those steps are done — the I/O path, the delimiter search, the number parse, the hash map, the threading,
the runtime — that drives the wall-clock number down, while the output stays byte-for-byte identical.

## Evaluation settings

The ranking metric is **wall-clock time** (m:ss.mmm) from process launch to last output byte, measured on
the fixed evaluation machine: eight cores of the AMD EPYC 7502P, 128 GB RAM, SMT and Turbo off, ext4, with
the input being 1,000,000,000 rows generated by the challenge's own generator (413 distinct stations in
the primary dataset; a separate 10,000-station run stresses the key-set ceiling). Each entry is launched
via its own shell script, which fixes the JDK build and JVM flags (several rungs use a GraalVM build or
`--enable-preview`); correctness is checked against the reference output before timing. The numbers each
rung reports are the published leaderboard results for that exact entry — not re-run here. Lower is better.
