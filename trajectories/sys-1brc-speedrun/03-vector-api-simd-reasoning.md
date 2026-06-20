05.979 seconds, down from nearly five minutes — the parallel mmap and the byte-keyed map did exactly the
two things I bet on: they filled all eight cores and killed the per-row allocation. So the cheap structural
wins are spent, and to find the next factor I have to look at what each core is *now* spending its time on,
line by line. The aggregation arithmetic is already trivial. The merge is a few hundred entries. What's
left, and what now dominates, is the scan: for every one of a billion lines, the previous loop walks the
name **one byte at a time** in a `while` loop testing each byte against `';'`, and it issues a separate
bounds-checked `bb.get` per byte. A station name averaging ~7 bytes means ~7 iterations, ~7 comparisons,
~7 branch checks, per line, a billion times — and each `bb.get` on a `MappedByteBuffer` carries a bounds
check. That byte-at-a-time delimiter search is the hot spot now. The number parse is already a handful of
integer ops, but the *finding* of the field boundary is linear in the name length and branch-heavy, and
branch mispredictions on variable-length names are not free.

The question is whether I can find the `';'` in a name without looping byte by byte — whether I can test
many bytes at once. The CPU can: it has SIMD lanes that compare 16 or 32 bytes against a constant in a
single instruction. Java now exposes this portably through the incubating Vector API. So instead of
scanning, I load a whole vector of bytes starting at the line and compare the entire vector against `';'`
in one operation: `line.compare(EQ, ';')` returns a mask, and `.toLong()` turns that into a bitmask whose
set bits mark exactly where the delimiters are. `Long.numberOfTrailingZeros(semicolons)` then gives me the
key length directly — no loop, no per-byte branch. I pick `SPECIES_256` when the hardware has 256-bit byte
lanes (32 bytes at once) and fall back to `SPECIES_128` otherwise; one vector op replaces the inner scan
for any name that fits in the vector. Most station names are short, so the common case is a single SIMD
compare per line.

To read bytes off the mapped file with this I move from `MappedByteBuffer` to `java.lang.foreign`:
`FileChannel.map` into a `MemorySegment` over a shared `Arena`, and `ByteVector.fromMemorySegment` loads a
vector straight out of mapped memory. The threads split the segment the same way as before
(`ceilDiv(byteSize, processorCount)` per thread, each thread adjusting its start forward to the next
newline via `findOffset`), and each runs its own map. So the parallel, per-thread, byte-keyed structure
survives — I'm only replacing the *inner scan* with vectorized compares.

There's a subtlety the vector forces me to handle: a name might be longer than the vector width. If
`line.compare(EQ, ';').toLong()` comes back zero, the `';'` isn't in the first 32 bytes, and I fall back to
a scalar loop from `keySize = vectorLength` onward until I find it — but that's the rare slow path
(`indexSimple`), and on the common short-name path the bitmask is nonzero and I never touch it. The tail of
each segment, where there isn't a full vector's worth of bytes left to safely over-read, also gets a simple
scalar parser (`parseDataPointSimple`) so I never read past the mapped region.

Now I can push the SIMD further, into the *hash and the key compare*, not just the delimiter search. For
the hash I don't want to fold in every byte of the name; I want a cheap, well-distributed hash from a fixed
amount of work. Since I already know the key length from the delimiter mask, I read the **first 4 bytes**
and the **last 4 bytes** of the name as two ints (`x`, `y`) and mix them with an FxHash-style step:
`rotateLeft(x * 0x9E3779B9, 5) ^ y) * 0x9E3779B9`. That's a constant two int loads and a couple of
multiplies regardless of name length — and for the short, distinct station names here, the first-and-last
4 bytes are plenty to separate keys. (When the key is shorter than 4 bytes I just read single bytes
instead.) I size the open-addressing table at `1<<17 = 128K` buckets, masking the hash with `CAPACITY-1`,
which keeps the load factor under the 10K ceiling tiny and probe chains short, exactly as before.

And the key *comparison* on a probe hit can be vectorized too. When I'm checking whether a bucket's stored
key equals the current name, I load the stored key as a `ByteVector` (`ByteVector.fromArray` over the
table's `keyData`), compare it lane-for-lane against the line vector, and check that the bytes up to the
delimiter all match. The clever bit is building the validity mask from the delimiter position itself:
`validMask = semicolons ^ (semicolons - 1)` is the run of low bits up to and including the first delimiter,
so `(eqMask & validMask) == validMask` is true exactly when every byte *before* the `';'` matched — a
single SIMD compare plus two integer ops replaces a byte-by-byte `Arrays.equals`. I keep the key length in
the node so I can reject mismatched-length keys before even comparing.

The number parse I keep as the branchless SWAR trick I'd already reached for, since it's the natural
scaled-integer parse and it's genuinely elegant: load 8 bytes of the value as one little-endian `long`,
locate the decimal point by the fact that the 4th bit of an ASCII digit is 1 while the `'.'`'s is 0
(`numberOfTrailingZeros(~word & 0x10101000)` gives the separator position, which is 12, 20, or 28
depending on the value's length), recover the sign from `(~word << 59) >> 63`, shift the digit bytes into
fixed positions, mask to `0x0F000F0F00`, and multiply by the magic constant
`100*0x1000000 + 10*0x10000 + 1` so that the three digit values land summed in bits 32–41 of the product —
`(product >>> 32) & 0x3FF` is `100*hundreds + 10*tens + units`, the temperature times ten. One load, no
branches, no per-digit loop. I apply the sign with `(absValue ^ signed) - signed`. The whole value parse is
a dozen integer instructions with zero branches, which matters because branchy parsing was part of what was
costing me.

So the loop per line is now: one vector load of the name region; one SIMD compare to find `';'` and read
off the key length; a fixed first-4/last-4-byte hash; a bucket probe whose hit-check is one SIMD key
compare; then a single-`long` branchless number parse; update min/max/sum/count in the node. Threads run
independent maps and merge into a `TreeMap` at the end, sorted, as before. The byte-at-a-time scan that
was the bottleneck is gone — replaced by data-parallel compares that chew through 32 bytes per
instruction.

The bar is the standing 05.979. The bet is that the dominant cost after parallelizing was the
byte-at-a-time, branch-heavy delimiter scan and key compare, and that replacing those with explicit SIMD —
finding the `';'`, hashing from fixed offsets, and comparing keys all in vector instructions — cuts the
per-line work enough to roughly halve the wall-clock, into the low-three-second range. The hedges are real:
the Vector API is incubating and its codegen quality varies, the long-name and tail slow paths add
branches I have to keep off the common path, and over-reading a vector's worth of bytes means I must guard
the segment tail carefully. But if the diagnosis is right that the scan was the new hot spot, vectorizing
it is the most direct attack, and it should land comfortably under 05.979. The full module is in the
answer.
