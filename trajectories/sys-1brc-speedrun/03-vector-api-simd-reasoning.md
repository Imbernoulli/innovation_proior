05.979 seconds, down from 04:49.679 — that is a 48.4× drop, and the two things I bet on both paid: the
eight cores filled and the per-row allocation died. Before I look for the next factor I want to read the
new number the way I read the last one, because the decomposition tells me what is now on the critical path.
289.679 seconds fell to 5.979, and I can split that 48.4× into its two independent causes. The threading
gave up to 8×. Whatever is left of the factor came from the constant-factor overhead removal, so the
implied per-row work improvement is `48.4 / 8 ≈ 6×`. I can check that against the wall-clock directly:
5.979 s across eight cores is about `5.979 × 8 = 47.8` core-seconds of CPU for a billion rows, i.e. ~47.8
nanoseconds of work per row, versus the baseline's ~290 ns per row on one core — a 6.1× shrink in per-row
work, matching the 6× I just inferred from the other direction. So the picture is consistent: roughly 8×
from cores, roughly 6× from killing allocation and decode, and now I have to attack the ~48 ns/row that
remains. On a 2.5 GHz core that is about 120 cycles per row.

Where do those 120 cycles go now? The aggregation arithmetic is trivial — a few cycles. The merge is a few
hundred entries, off the hot path entirely. What is left, and what now dominates, is the *scan*. For every
one of a billion lines, the previous loop walks the station name **one byte at a time** in a `while` loop,
testing each byte against `';'` and issuing a separate bounds-checked `bb.get` per byte. A station name
averaging ~7 bytes means ~7 iterations, ~7 comparisons, ~7 branch checks, and ~7 bounds checks per line, a
billion times over. Each `bb.get` on a `MappedByteBuffer` carries a bounds check; the loop's exit condition
is data-dependent on the name length, so the branch predictor cannot reliably guess when the name ends; and
a misprediction on a tight loop costs on the order of fifteen cycles. Put ~7 iterations at several cycles
each together with the occasional misprediction and the delimiter search alone is plausibly 25–35 of those
120 cycles. That byte-at-a-time scan is the bottleneck, and the number parse — already a handful of integer
ops in the previous rung — is not what is left to cut; the *finding* of the field boundary is.

So the question is sharp: can I find the `';'` in a name without looping byte by byte — can I test many
bytes at once? The CPU can. It has SIMD lanes that compare 16 or 32 bytes against a constant in a single
instruction, and Java now exposes this portably through the incubating Vector API. Instead of scanning, I
load a whole vector of bytes starting at the line and compare the entire vector against `';'` in one
operation: `line.compare(EQ, ';')` returns a lane mask, and `.toLong()` turns that into a bitmask whose set
bits mark exactly where the delimiters are. Then `Long.numberOfTrailingZeros(semicolons)` gives me the key
length *directly* — no loop, no per-byte branch, one instruction where there were seven iterations.

The vector width is not a free parameter, and picking it wrong would waste the whole move, so let me pin it
to the hardware. The evaluation machine is an AMD EPYC 7502P, which is Zen2. Zen2 implements AVX2 —
256-bit vectors — and has no AVX-512 at all. So 256 bits, i.e. 32 bytes, is the widest lane the machine can
compare in a single native instruction; asking for `SPECIES_512` would make the JVM decompose each op into
two 256-bit halves with no throughput gain and extra overhead. That is exactly why I choose the species by
querying the hardware: `SPECIES_PREFERRED.length() >= 32 ? SPECIES_256 : SPECIES_128`. On this box the
preferred length is 32, so I get `SPECIES_256` — 32 bytes per compare, the machine's native width. Most
station names are far shorter than 32 bytes, so the common case is a single SIMD compare that spans the
entire name and then some.

To read bytes off the mapped file into a vector I move from `MappedByteBuffer` to `java.lang.foreign`:
`FileChannel.map` into a `MemorySegment` over a shared `Arena`, and `ByteVector.fromMemorySegment` loads a
vector straight out of mapped memory with no intermediate copy. The parallel, per-thread, byte-keyed
structure from the previous rung survives unchanged — the threads still split the file
(`ceilDiv(byteSize, processorCount)` per thread, each adjusting its start forward to the next newline via
`findOffset`), each still runs its own open-addressed map, and the merge is still a sorted `TreeMap`. I am
replacing only the *inner scan*, which is the thing profiling puts on the critical path.

The move to `MemorySegment` is forced by the tool rather than chosen for taste, and it is worth being clear
why. `ByteVector.fromMemorySegment` loads its lanes from a `MemorySegment`; it cannot take a
`MappedByteBuffer` directly, so to feed the Vector API off mapped memory I have to map the file as a
`MemorySegment` in the first place. I do that over a single shared `Arena` — one mapping for the whole file,
handed to all eight threads, each reading its own newline-aligned slice — so there is no per-thread remap
and no copy; the threads read straight out of the one mapped region into vector registers. That the
foreign-memory API and the Vector API are designed to compose this way is exactly what makes the
32-bytes-per-instruction scan reach all the way down to the page cache with nothing in between.

The vector forces two guarded slow paths, and I want to be explicit that they are rare enough to keep off
the common path. First, a name longer than the vector: if `line.compare(EQ, ';').toLong()` comes back zero,
the `';'` is not in the first 32 bytes, so I fall back to a scalar loop from `keySize = vectorLength` onward
(`indexSimple`). A station name is at most 100 bytes but here averages ~7 and rarely exceeds a dozen, so a
name over 32 bytes is a small minority and this branch almost never fires; on the common short-name path the
bitmask is nonzero and I never touch it. Second, the segment tail: loading 32 bytes at a line start reads
*past* the name into the value and the next line, which is fine because I mask down to the key length — but
near the end of a segment, over-reading 32 bytes could run past the mapped region and fault. So the last
vector's worth of bytes in each segment gets a simple scalar parser (`parseDataPointSimple`). That is 32
bytes of caution out of a ~1.5 GB segment, a vanishing fraction of the work, spent to make the over-read
safe. The same tail guard covers the number parse, which reads a full 8-byte `long` at `offset + keySize +
1` and therefore also over-reads up to 8 bytes past a value near the segment end; because the scalar tail
path handles the final stretch, neither the name vector load nor the value `long` load ever faults off the
end of the mapping.

Now that the scan no longer visits every byte, the *hash* has to change, and thinking about why is what
picks the new hash. In the previous rung I folded `hash = 31*hash + b` for free during the byte-at-a-time
walk. But there is no byte-at-a-time walk anymore — the SIMD compare skips straight to the key length — so
folding a per-byte polynomial would mean adding back a separate pass over the name purely to hash it, which
is exactly the linear-in-length cost I just paid to remove. Instead I want a hash whose cost is *fixed*
regardless of name length. Since the delimiter mask already handed me the key length, I read the **first 4
bytes** and the **last 4 bytes** of the name as two ints `x` and `y` and mix them with an FxHash-style step:
`rotateLeft(x * 0x9E3779B9, 5) ^ y`, then `* 0x9E3779B9`. The constant `0x9E3779B9` is not arbitrary — it is
`2^32` divided by the golden ratio (≈ 2654435769), the Fibonacci-hashing multiplier chosen precisely because
multiplying by it spreads input bits across the whole word with good avalanche, so even the short, similar
byte windows of station names diffuse into well-separated bucket indices. That is a constant two int-loads and a couple of
multiplies for any name length. When the name is shorter than 4 bytes I read single bytes instead so I do
not read across the delimiter. The worry with a fixed-window hash is collisions — two distinct names sharing
their first and last four bytes — so let me be careful about what that costs. It costs *nothing in
correctness*: on a probe I still do a full key comparison, so a hash collision only sends me one slot
further down the probe chain, never to a wrong answer. It costs only probe length, and for a few hundred
short, distinct station names the first-and-last-four-bytes window separates them well, so probe chains stay
at essentially one. The 128K-bucket table (`1 << 17`) masked with `CAPACITY - 1` keeps the load factor tiny
against the 10K ceiling, exactly as before.

The key *comparison* on a probe hit can be vectorized too, and there is a neat bitmask trick that makes it
one compare. When I check whether a bucket's stored key equals the current name, I load the stored key as a
`ByteVector` (`ByteVector.fromArray` over the table's `keyData`) and compare it lane-for-lane against the
line vector, producing `eqMask`. But I must only require the bytes *before* the `';'` to match — the lanes
past the delimiter hold value bytes and next-line garbage that I do not care about. I build the validity
mask straight from the delimiter position: `validMask = semicolons ^ (semicolons - 1)`. Let me verify that
identity does what I claim. If the lowest set bit of `semicolons` is at position `k` (the first `';'` is in
lane `k`), then `semicolons - 1` clears that bit and sets all bits below it, so the XOR lights up exactly
bits `0..k` — a run of `k+1` low bits. Concretely, if the `';'` is in lane 3, `semicolons = 0b1000 = 8`,
`semicolons - 1 = 7`, and `8 ^ 7 = 0b1111 = 15`, marking lanes 0 through 3. Then `(eqMask & validMask) ==
validMask` is true exactly when every one of those low lanes matched — a single SIMD compare plus two
integer ops in place of a byte-by-byte `Arrays.equals`. I also store the key length in the node and reject a
length mismatch before comparing at all, so different-length names never reach the vector compare.

The number parse I keep as the branchless SWAR trick I reached for in the previous rung, but now in its full
magic-constant form, because it is the natural scaled-integer parse and genuinely elegant. I load 8 bytes of
the value as one little-endian `long` and locate the decimal point by a bit fact: the 4th bit (`0x10`) of an
ASCII digit is 1, while the 4th bit of `'.'` is 0. Check it — `'5'` is `0x35`, and `0x35 & 0x10 = 0x10`,
bit set; `'.'` is `0x2E`, and `0x2E & 0x10 = 0`, bit clear. So `numberOfTrailingZeros(~word & 0x10101000)`
finds the byte position of the separator (12, 20, or 28 bits in, depending on whether the value is one or
two integer digits and whether there is a minus). I recover the sign from `(~word << 59) >> 63`, an
arithmetic right shift that yields all-ones (`-1`) if the low byte was `'-'` and `0` otherwise. I shift the
digit bytes into fixed positions, mask to `0x0F000F0F00` to isolate the three digit nibbles, and multiply
by a magic constant `100*0x1000000 + 10*0x10000 + 1`. Let me confirm that constant: `0x64000000 +
0x000A0000 + 0x00000001 = 0x640A0001`, so the multiply is by `0x640a0001`, which lines up `hundreds`,
`tens`, and `units` so that `100*h + 10*t + u` lands *summed* in bits 32–41 of the product. Then `(product
>>> 32) & 0x3FF` reads it off — and `0x3FF` is 10 bits, holding up to 1023, comfortably above the maximum
scaled value 999 (99.9 × 10), so nothing overflows the mask. I apply the sign with `(absValue ^ signed) -
signed`. One load, no branches, no per-digit loop, which matters because branchy parsing is part of what
this rung is trying to strip.

I want to check that the fixed `shift = 28 - decimalSepPos` really does align every value shape onto the
same digit positions, since that is the whole reason one magic multiply can serve all four shapes. Walk the
separator byte index through the cases. `"2.5"` puts `'.'` at byte 1, so the separator sits at bit `8*1 + 4
= 12`, and `shift = 28 - 12 = 16`. `"12.3"` puts `'.'` at byte 2, bit `8*2 + 4 = 20`, `shift = 8`.
`"-12.3"` puts `'.'` at byte 3, bit `28`, `shift = 0`. So `decimalSepPos ∈ {12, 20, 28}` maps to `shift ∈
{16, 8, 0}`, and each shift slides that shape's digit bytes up to the *same* fixed lanes the mask
`0x0F000F0F00` and the multiply expect — a one-digit value is pushed the furthest, a three-integer-digit
signed value not at all, and they meet in the middle. The return advance uses the same position: `offset +
(decimalSepPos >>> 3) + 3` steps to the next line, where `decimalSepPos >>> 3` is the `'.'` byte index.
Check it lands right: `"2.5\n"` is four bytes and `'.'` is at index 1, so `1 + 3 = 4` steps exactly past
the newline; `"12.3\n"` is five bytes, `'.'` at index 2, `2 + 3 = 5`, past the newline; `"-12.3\n"` is six
bytes, `'.'` at index 3, `3 + 3 = 6`, past the newline. All three land on the start of the next record, so
the same branchless expression both parses the value and advances the cursor for every shape. And the value
range is safe at the extremes: `99.9` scales to `999`, `-99.9` to `-999`, both inside the 10-bit `0x3FF`
mask and inside a signed range, so nothing the data can throw at the parser overflows it.

So the per-line loop is now: one vector load of the name region; one SIMD compare to find `';'` and read off
the key length; a fixed first-4/last-4-byte hash; a bucket probe whose hit-check is one SIMD key compare;
then a single-`long` branchless number parse; update `min/max/sum/count` in the node. Threads run
independent maps and merge into a sorted `TreeMap` at the end. The byte-at-a-time scan that was the
bottleneck is gone, replaced by data-parallel compares that chew through 32 bytes per instruction.

Let me weigh whether SIMD is really the right lever for this step against the obvious alternatives, because I
want the choice to be forced rather than fashionable. More threads is not available — I am already pinned to
eight cores and using them. A faster map is not the lever — the probe chains are already essentially length
one at 0.3–7.6% occupancy, so there is almost nothing to win there. A better number parse is not the lever —
it is already a dozen integer ops and, by my per-row accounting, a small slice of the 120 cycles. The one
part of the loop that is still linear in the name length and branch-heavy is the delimiter scan, and that is
exactly what the profile puts on top. So vectorizing the scan is the forced move, not a shiny one.

There is a design choice inside "vectorize" worth making explicit, because a 64-bit register can also test
several bytes at once. I am, after all, already using exactly that for the number parse: a `long` holds 8
bytes and bit-twiddling finds a target byte within it. So why reach for the wide Vector API on the *name*
rather than the same register-width trick? Because a name can be up to 32 bytes of interest and averages
~7, and a single 256-bit compare spans the whole name — and then some — in one native instruction, reading
the key length straight off the mask. A 64-bit word covers only 8 bytes, so a name spilling past 8 bytes
would need a second word and extra logic to stitch the delimiter position across the boundary. For the name,
the wide vector is the more direct expression of "compare the entire key at once," so I use it here and keep
the register-width trick for the fixed 8-byte value where it is the natural fit. That is the choice this rung
makes; whether the wide vector's incubator codegen actually beats the register-width alternative on this
hardware is the kind of thing only the wall-clock will settle, and I hold it as an open question rather than
a settled one.

Let me re-budget the per-row cycles under this plan so the target is arithmetic rather than hope. The
previous rung's ~47.8 ns/row split roughly into the delimiter scan (~11 ns, the ~7-byte byte-at-a-time
loop), a byte-by-byte key compare on the probe hit (several more ns, another linear-in-length walk over the
name), the fixed hash, the map probe with its potential cache-miss stall, the number parse, and the
aggregate update. This rung collapses *two* of those linear-in-length costs at once — the scan and the key
compare both become single 32-byte SIMD compares — and makes the hash fixed-cost. If that removes on the
order of ~18–20 ns/row of the ~48, per-row work falls toward ~28–30 ns, and `5.979 × (29/48) ≈ 3.6`
seconds. That is the low-three-second landing I am aiming at, and it is why "roughly halve" is the honest
expectation: I am removing about half of the per-row work, not all of it, because the map probe, the number
parse, and the update remain.

The bar is the standing 05.979. The bet is that the dominant cost after parallelizing was the
byte-at-a-time, branch-heavy delimiter scan (and the byte-by-byte key compare), and that replacing both with
explicit SIMD — one compare to find the `';'`, a fixed-cost hash, one compare to check the key — cuts the
per-line work enough to roughly halve the wall-clock, into the low-three-second range. The falsifiable form,
in the metric: if the scan really was ~25–35 of the ~120 cycles per row, removing most of it should land me
around 3 seconds; if instead I only shave a little, then either the incubator's codegen is not folding the
vector path into tight machine code, or the true bottleneck was something I mis-attributed, like
memory-latency stalls on the map probe rather than the scan. The honest hedges are real: the Vector API is
incubating and its generated code quality varies from JDK to JDK; the long-name and tail slow paths add
branches I must keep strictly off the common path; and over-reading a vector's worth of bytes means I have
to guard the segment tail so I never fault. There is a second falsifiable prediction I can make with the two
datasets the harness runs: because the hash is fixed-cost and the probe chains stay short even at the
10,000-station ceiling's 7.6% load factor, the vector path should not regress much between the 413-station
primary run and the 10,000-station stress run — if the stress run blew up disproportionately, that would
tell me the fixed first-4/last-4 hash is colliding badly on the larger key set and lengthening probes, which
would send me back to the hash rather than the scan. But if the diagnosis is right that the scan is the new hot spot,
vectorizing it is the most direct attack, and it should land comfortably under 05.979. The full module is in
the answer.
