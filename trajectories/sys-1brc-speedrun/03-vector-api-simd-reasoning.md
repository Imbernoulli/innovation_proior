05.979 seconds, down from 04:49.679 — a 48.4× drop, and the two things I bet on both paid: the eight cores
filled and the per-row allocation died. Read the new number the way I read the last one. 289.679 s fell to
5.979, and I can split that 48.4× into its two independent causes: threading gave up to 8×, so the implied
per-row work improvement is `48.4 / 8 ≈ 6×`. That checks against the wall-clock directly — 5.979 s across
eight cores is ~47.8 core-seconds for a billion rows, ~47.8 ns/row, versus the baseline's ~290 ns/row on one
core, a 6.1× shrink in per-row work matching the 6× inferred from the other direction. So roughly 8× from
cores, 6× from killing allocation and decode, and now I attack the ~48 ns/row that remains, about 120 cycles
per row on a 2.5 GHz core.

Where do those 120 cycles go now? The aggregation is a few cycles; the merge is off the hot path. What
dominates is the *scan*: for every one of a billion lines, the previous loop walks the name **one byte at a
time**, testing each against `';'` with a separate bounds-checked `bb.get`. A ~7-byte name means ~7
iterations, ~7 comparisons, ~7 bounds checks per line, a billion times. The loop's exit condition is
data-dependent on the name length, so the branch predictor cannot reliably guess when the name ends, and a
misprediction on a tight loop costs ~15 cycles. Put ~7 iterations at several cycles each together with the
occasional misprediction and the delimiter search alone is plausibly 25–35 of those 120 cycles. That
byte-at-a-time scan is the bottleneck — not the number parse, which is already a handful of integer ops, but
the *finding* of the field boundary.

So the question is sharp: can I test many bytes at once instead of looping byte by byte? The CPU can, and
Java exposes it through the incubating Vector API. Instead of scanning, I load a whole vector of bytes at the
line start and compare the entire vector against `';'` in one operation: `line.compare(EQ, ';')` returns a
lane mask, `.toLong()` turns it into a bitmask whose set bits mark the delimiters, and
`Long.numberOfTrailingZeros(semicolons)` gives me the key length *directly* — one instruction where there
were seven iterations. The vector width is pinned to the hardware: the evaluation machine is an AMD EPYC
7502P, Zen2, which implements AVX2 (256-bit) and has no AVX-512, so 32 bytes is the widest single-instruction
compare; `SPECIES_512` would decompose into two 256-bit halves for no gain. I choose the species by querying
the hardware — `SPECIES_PREFERRED.length() >= 32 ? SPECIES_256 : SPECIES_128` — which yields `SPECIES_256`
here. Most names are far shorter than 32 bytes, so the common case is a single compare spanning the entire
name and then some.

`ByteVector.fromMemorySegment` loads its lanes from a `MemorySegment`, not a `MappedByteBuffer`, so to feed
the Vector API off mapped memory I must map the file as a `MemorySegment` in the first place — over a single
shared `Arena`, one mapping for the whole file handed to all eight threads, each reading its own
newline-aligned slice with no per-thread remap and no copy. That the foreign-memory and Vector APIs compose
this way is what lets the 32-bytes-per-instruction scan reach straight down to the page cache. The parallel,
per-thread, byte-keyed structure from the previous rung survives unchanged — threads still split the file
and adjust each start forward to the next newline, each still runs its own open-addressed map, and the merge
is still a sorted `TreeMap`. I am replacing only the inner scan.

The vector forces two guarded slow paths, and both are rare enough to keep off the common path. If
`line.compare(EQ, ';').toLong()` comes back zero the `';'` is not in the first 32 bytes, so I fall back to a
scalar loop from `keySize = vectorLength` onward — a name over 32 bytes is a small minority here, where names
average ~7 and rarely exceed a dozen. And near the end of a segment, loading 32 bytes at a line start could
over-read past the mapped region and fault, so the last vector's worth of bytes in each segment gets a scalar
parser — 32 bytes of caution out of a ~1.5 GB segment. The same tail guard covers the number parse, which
reads a full 8-byte `long` and therefore also over-reads up to 8 bytes near the segment end.

Now that the scan no longer visits every byte, the *hash* has to change, and thinking about why picks the new
hash. In the previous rung I folded `hash = 31*hash + b` for free during the byte-at-a-time walk. But there
is no byte-at-a-time walk anymore, so folding a per-byte polynomial would add back a separate pass over the
name purely to hash it — the linear-in-length cost I just paid to remove. Instead I want a hash whose cost is
*fixed* regardless of length. Since the delimiter mask already handed me the key length, I read the first 4
and last 4 bytes of the name as two ints `x` and `y` and mix them FxHash-style: `rotateLeft(x * 0x9E3779B9,
5) ^ y`, then `* 0x9E3779B9`. The constant is `2^32` over the golden ratio, the Fibonacci-hashing multiplier
chosen because multiplying by it spreads input bits across the whole word with good avalanche, so even short,
similar byte windows diffuse into well-separated buckets. When the name is shorter than 4 bytes I read single
bytes so I do not read across the delimiter. The worry with a fixed-window hash is collisions — two names
sharing their first and last four bytes — but that costs *nothing in correctness*: on a probe I still do a
full key comparison, so a collision only sends me one slot further down the chain. It costs only probe
length, and for a few hundred short, distinct names the first-and-last-four window separates them well, so
chains stay at essentially one. The 128K-bucket table masked with `CAPACITY - 1` keeps the load factor tiny
against the 10K ceiling, as before.

The key *comparison* on a probe hit vectorizes too. I load the bucket's stored key as a `ByteVector` and
compare it lane-for-lane against the line vector, producing `eqMask`, but I must only require the bytes
*before* the `';'` to match — the lanes past the delimiter hold value bytes and next-line garbage. I build
the validity mask straight from the delimiter position: `validMask = semicolons ^ (semicolons - 1)`, which
lights up exactly bits `0..k` when the first `';'` is in lane `k` (for lane 3, `8 ^ 7 = 15`, marking lanes
0–3). Then `(eqMask & validMask) == validMask` is true exactly when every low lane matched — one SIMD
compare plus two integer ops in place of a byte-by-byte `Arrays.equals`. I also store the key length and
reject a length mismatch before comparing, so different-length names never reach the vector compare.

The number parse I keep as the branchless SWAR trick from the previous rung, now in its full magic-constant
form. I load 8 bytes of the value as one little-endian `long` and locate the decimal point by a bit fact: the
4th bit (`0x10`) of an ASCII digit is 1 while the 4th bit of `'.'` is 0 (`'5'` is `0x35`, `0x35 & 0x10` set;
`'.'` is `0x2E`, `0x2E & 0x10` clear), so `numberOfTrailingZeros(~word & 0x10101000)` finds the separator's
byte position. I recover the sign from `(~word << 59) >> 63` (all-ones if the low byte was `'-'`, else 0),
shift the digit bytes into fixed positions by `shift = 28 - decimalSepPos`, mask to `0x0F000F0F00` to isolate
the three digit nibbles, and multiply by `0x640a0001` (= `100*0x1000000 + 10*0x10000 + 1`), which lines up
hundreds/tens/units so that `100*h + 10*t + u` lands summed in bits 32–41; `(product >>> 32) & 0x3FF` reads
it off, the 10-bit mask comfortably holding the maximum scaled value 999. Sign is applied with `(absValue ^
signed) - signed`. The fixed shift is what lets one magic multiply serve all four shapes: `decimalSepPos ∈
{12, 20, 28}` for one-digit, two-digit, and signed-two-digit values maps to `shift ∈ {16, 8, 0}`, sliding
each shape's digits onto the same fixed lanes; and the return advance `offset + (decimalSepPos >>> 3) + 3`
steps exactly past the newline for every shape. `99.9` and `-99.9` scale to `±999`, inside the mask and the
signed range, so nothing the data can throw at the parser overflows it. One load, no branches, no per-digit
loop.

So the per-line loop is now one vector load of the name region; one SIMD compare to find `';'` and read off
the key length; a fixed first-4/last-4 hash; a bucket probe whose hit-check is one SIMD key compare; then a
single-`long` branchless number parse; update `min/max/sum/count`. The byte-at-a-time scan that was the
bottleneck is gone.

SIMD is the forced lever here, not a fashionable one: more threads is not available (already pinned to
eight), a faster map is not the lever (probe chains are already length one), a better number parse is not the
lever (already a dozen integer ops) — the one part still linear in name length and branch-heavy is the
delimiter scan, exactly what the profile puts on top. There is a choice *inside* "vectorize," though, since a
64-bit register can also test several bytes at once — I use exactly that for the number parse. Why the wide
Vector API on the *name* rather than the register-width trick? Because a name can be up to 32 bytes and
averages ~7, and a single 256-bit compare spans the whole name in one native instruction, reading the key
length straight off the mask; a 64-bit word covers only 8 bytes and would need a second word plus logic to
stitch the delimiter position across the boundary. For the name, the wide vector is the more direct
expression of "compare the entire key at once"; I keep the register-width trick for the fixed 8-byte value.
Whether the incubator's wide-vector codegen actually beats the register-width alternative on this hardware is
something only the wall-clock will settle, and I hold it as an open question.

Re-budget the per-row cycles so the target is arithmetic rather than hope. The ~47.8 ns/row split roughly
into the delimiter scan (~11 ns, the byte-at-a-time loop), a byte-by-byte key compare on the probe hit
(another linear-in-length walk), the fixed hash, the map probe with its potential cache-miss stall, the
number parse, and the update. This rung collapses *two* of those linear-in-length costs at once — the scan
and the key compare both become single 32-byte SIMD compares — and makes the hash fixed-cost. If that
removes ~18–20 ns/row of the ~48, per-row work falls toward ~28–30 ns, and `5.979 × (29/48) ≈ 3.6` seconds
is the low-three-second landing I aim at — "roughly halve," because the map probe, the number parse, and the
update remain. The falsifiable form: if the scan really was ~25–35 of the ~120 cycles, removing most of it
should land me around 3 seconds; if I only shave a little, then either the incubator's codegen is not folding
the vector path into tight machine code, or the true bottleneck was memory-latency stalls on the map probe
rather than the scan. A second a-priori prediction the two datasets let me make: because the hash is
fixed-cost and probe chains stay short even at the 10,000-station 7.6% load factor, the vector path should
not regress much between the primary and stress runs — if the stress run blew up disproportionately, the
fixed first-4/last-4 hash is colliding and lengthening probes, which would send me back to the hash rather
than the scan. The honest hedges: the Vector API is incubating with uneven codegen, the slow paths add
branches I must keep strictly off the common path, and over-reading a vector's worth of bytes means guarding
the segment tail. The full module is in the answer.
