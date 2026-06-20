The baseline came back at 04:49.679 — nearly five minutes — and the profile is exactly what I predicted
when I wrote it: one core busy, seven idle, and a torrent of short-lived allocation underneath. So I have
two big multipliers sitting untouched, and I should take both at once because they compound. The machine
has eight cores and I'm using one; that alone is up to an 8× ceiling. And every line is paying for a
decoded `String`, a `String[]` from `split`, and a boxed `Double` — billions of objects whose only purpose
is to be immediately discarded, which means the GC is running flat out and the parse is doing far more
work than the data shape requires. The baseline's time is dominated by *overhead per row*, not by the
actual min/max/sum arithmetic, so the move is to attack the per-row overhead and the single-threadedness
together.

Start with the threading, because it sets the shape of everything else. To use eight cores I have to give
each one an independent slice of the file and let it build its own aggregate, then merge the eight
aggregates at the end — which works precisely because min/max/sum/count is associative, the same property
the baseline's combiner already relied on. So I'll split the file into `numberOfSegments = availableProcessors()`
segments. But I can't split on byte offsets blindly: a naive cut would land in the middle of a line, and
one segment would see a truncated `station;temp` while its neighbour sees the other half. The fix is to
choose each segment's real boundary by starting from the nominal offset and scanning forward to the next
`'\n'` — so every segment begins exactly at a line start and ends at a line end. I compute the nominal
`segmentSize = fileSize / numberOfSegments`, then for boundary *i* seek to `i * segmentSize` and walk to
the next newline; the previous segment's end is this segment's start, so there is no overlap and no gap.
(For a tiny file the split is pointless, so below a threshold I just use one segment.) Each segment then
runs on its own, and a `parallel()` stream over the segments fans them across the cores.

Now the per-row overhead, which is the bigger structural change. The baseline reads the file as *decoded
lines*; I want to stop decoding entirely and work on raw bytes. `FileChannel.map(READ_ONLY, start, len)`
maps a segment of the file straight into memory as a `MappedByteBuffer`, so I read bytes out of the page
cache with no `read()` syscalls, no per-line `String` allocation, and no UTF-8 decode of all 12 GB. The
inner loop becomes: from the current position, walk byte by byte accumulating the station name into a
scratch `byte[100]` until I hit `';'`, then parse the temperature, then advance past the newline. No
`split`, no intermediate objects — just a pointer marching through mapped memory.

Two pieces of that loop deserve their own thought. First, the number parse. `Double.parseDouble` is a
general float parser, but the data is far more constrained than a general float: it is always one of
`D.D`, `DD.D`, `-D.D`, `-DD.D` — one optional minus, one or two integer digits, a dot, exactly one
fractional digit. I never need a `double` at all during parsing; I can read it as a scaled integer. So I
peek at the bytes directly: if the first byte is `'-'`, set a sign of −1 and step past it; then if the
byte two positions ahead is `'.'` the integer part is one digit and the value is `d0*10 + d1` (the digit
after the dot); otherwise the integer part is two digits and it is `d0*100 + d1*10 + d2`. Each digit is
just `byte - '0'`. That turns the whole parse into a handful of integer adds and multiplies with no
function call and no floating point, and I divide by ten only when I finally store `temp / 10.0`. (Even the
inner unroll — checking the `'.'` position to branch between the one- and two-digit-integer cases — is the
kind of thing the leaderboard shares; the idea of unrolling it this far I'm borrowing from another entry's
trick, but it falls straight out of the data's fixed shape.)

Second, and this is where the real allocation savings live: the per-station lookup. The baseline hashes a
`String` key into a `HashMap` for every row. I never want to materialize a `String` per row — the station
name is a range of bytes in my scratch buffer, and I want to look it up *as bytes*. So I build a small
custom map specialized to this task: a flat open-addressing table. I compute a hash of the name bytes as I
scan them (`hash = 31*hash + b`, the same polynomial Java's `String.hashCode` uses, accumulated for free
during the delimiter scan), and use `hash & (MAPSIZE - 1)` to pick a slot in a power-of-two-sized array
(`MAPSIZE = 1<<17 = 128K` slots, comfortably above the 10,000-station ceiling so the table stays sparse
and probe chains stay short). On a hit, I update the slot's `Result` (min/max/sum/count) in place. On a
miss (null slot, or a slot whose stored key bytes don't match), I linear-probe forward — `slot = (slot+1) &
(MAPSIZE-1)` — until I find the matching key or an empty slot, then create the `Result` and copy the name
bytes in *once*. Key comparison is `Arrays.equals` over the byte ranges, so I compare bytes directly and
never decode. This is the move that kills the billions of allocations: a `String` is created exactly once
per *distinct station* (a few hundred), at merge time, instead of once per *row*.

Open addressing with linear probing is the right structure here rather than chaining: there are no per-node
list objects, everything lives in two flat arrays (`Result[] slots` and `byte[][] keys`) so the table is
cache-friendly, and with the table 100× larger than the key set the load factor is tiny and collisions are
rare. Each segment-thread gets its own `ByteArrayToResultMap`, so there is zero contention during the hot
loop — no locks, no shared state — and only at the very end do I drain each thread's map into a list of
`(byte[] key, Result)` entries, decode each key to a `String` exactly once, and `Collectors.toMap` them
into a `TreeMap` with a merge function that combines colliding stations across threads. The `TreeMap`
keeps the output sorted, same as the baseline.

So the structure is: split the file at newline-aligned boundaries into one segment per core; `parallel()`
over the segments; each thread memory-maps its segment and runs a byte-pointer loop that scans the name to
`';'` (hashing as it goes), parses the temperature as a scaled integer, and updates its own open-addressed
byte-keyed map in place; merge the per-thread maps into a sorted `TreeMap` and print. Every layer the
baseline paid for per row — the decoded `String`, the `split` array, the boxed `Double`, the `String` hash,
the single core — is gone or amortized.

Setting the bar: the baseline is 04:49.679. The two changes I'm stacking are an up-to-8× win from using
all the cores and a large constant-factor win from killing per-row allocation, the UTF-8 decode, the
`split`, and the float parse. Those multiply, so I should not be landing at "a bit under five minutes" — I
should be landing in *seconds*. The honest hedge is that I won't get the full 8× from threads (the merge,
the page-fault-driven I/O as the mmap warms, and imbalance between segments all cost something) and the
custom map has to actually keep probe chains short for the constant-factor win to materialize. But if the
diagnosis is right — that the baseline's time was almost entirely per-row overhead and idle cores — then
removing both should drop it by roughly two orders of magnitude, into the single-digit-seconds range. The
full module is in the answer.
