The parallel mmap rung landed at **00:05.979** — about 48× over the baseline — which means the two cheap structural wins are spent: all eight cores are filled and the per-row allocation is gone. To find the next factor I have to look at what each core now spends its time on, line by line. The aggregation arithmetic is already trivial and the merge is a few hundred entries. What dominates now is the *scan*: for every one of a billion lines, the previous loop walks the name **one byte at a time** in a `while` loop, testing each byte against `';'` and issuing a separate bounds-checked `bb.get` per byte. A ~7-byte name means ~7 iterations, ~7 comparisons, ~7 branch checks per line, a billion times — and branch mispredictions on variable-length names are not free. The byte-at-a-time delimiter search is the bottleneck.

The method is **explicit SIMD via the JDK Vector API** (the merykitty approach): test many bytes at once instead of looping. The CPU has SIMD lanes that compare 16 or 32 bytes against a constant in a single instruction, and Java exposes them portably through `jdk.incubator.vector`. So instead of scanning, I load a whole vector of bytes starting at the line and compare it against `';'` in one operation: `line.compare(EQ, ';')` returns a lane mask, and `.toLong()` turns it into a bitmask whose set bits mark exactly where the delimiters are. Then `Long.numberOfTrailingZeros(semicolons)` gives the key length *directly* — no loop, no per-byte branch. I pick `SPECIES_256` (32 bytes at once) when the hardware has 256-bit byte lanes and fall back to `SPECIES_128` otherwise, so one vector op replaces the inner scan for any name that fits in the vector. Most names are short, so the common case is a single SIMD compare per line. To read bytes off the mapped file this way I move from `MappedByteBuffer` to `java.lang.foreign`: `FileChannel.map` into a `MemorySegment` over a shared `Arena`, and `ByteVector.fromMemorySegment` loads a vector straight out of mapped memory. The parallel, per-thread, byte-keyed structure survives unchanged; only the inner scan is replaced.

The vector forces two guarded slow paths. If `line.compare(EQ, ';').toLong()` comes back zero, the `';'` is not in the first 32 bytes, so I fall back to a scalar loop from `keySize = vectorLength` onward (`indexSimple`); on the common short-name path the bitmask is nonzero and I never touch it. The tail of each segment, where there isn't a full vector's worth of bytes left to safely over-read, also gets a simple scalar parser (`parseDataPointSimple`) so I never read past the mapped region.

I push SIMD further, into the *hash* and the *key compare*, not just the delimiter search. For the hash I want cheap, well-distributed mixing from a fixed amount of work regardless of name length. Since the delimiter mask already gave me the key length, I read the **first 4 bytes** and the **last 4 bytes** of the name as two ints `x` and `y` and mix them with an FxHash-style step: $\text{rotateLeft}(x \cdot \mathtt{0x9E3779B9},\,5) \oplus y$, then $\times \mathtt{0x9E3779B9}$. That is a constant two int-loads and a couple of multiplies for any length, and for short distinct station names the first and last four bytes separate keys well. (Below 4 bytes I read single bytes instead.) The open-addressing table is `1<<17 = 128K` buckets masked with `CAPACITY-1`, keeping the load factor tiny against the 10K ceiling and probe chains short, as before.

The key *comparison* on a probe hit is vectorized too. I load the bucket's stored key as a `ByteVector` (`ByteVector.fromArray` over the table's `keyData`) and compare it lane-for-lane against the line vector. The clever part is building the validity mask from the delimiter position itself: $\text{validMask} = \text{semicolons} \oplus (\text{semicolons} - 1)$ is the run of low bits up to and including the first delimiter, so $(\text{eqMask}\ \&\ \text{validMask}) = \text{validMask}$ is true exactly when every byte *before* the `';'` matched. A single SIMD compare plus two integer ops replaces a byte-by-byte `Arrays.equals`, and I keep the key length in the node to reject mismatched-length keys before comparing at all.

The number parse I keep as the branchless **SWAR** trick already reached for in the previous rung, since it is the natural scaled-integer parse and is genuinely elegant. Load 8 bytes of the value as one little-endian `long`; locate the decimal point by the fact that the 4th bit of an ASCII digit is 1 while the `'.'`'s is 0, so $\text{numberOfTrailingZeros}(\lnot\text{word}\ \&\ \mathtt{0x10101000})$ gives the separator position (12, 20, or 28 depending on the value's length); recover the sign from $(\lnot\text{word} \ll 59) \gg 63$ (an arithmetic shift yielding $-1$ if `'-'`, else $0$); shift the digit bytes into fixed positions; mask to `0x0F000F0F00`; and multiply by the magic constant $100\cdot\mathtt{0x1000000} + 10\cdot\mathtt{0x10000} + 1 = \mathtt{0x640a0001}$ so the three digit values land *summed* in bits 32–41 of the product. Then $(\text{product} \ggg 32)\ \&\ \mathtt{0x3FF}$ is $100h + 10t + u$, the temperature times ten, and I apply the sign with $(\text{absValue} \oplus \text{signed}) - \text{signed}$. One load, no branches, no per-digit loop.

So the per-line loop is now: one vector load of the name region; one SIMD compare to find `';'` and read off the key length; a fixed first-4/last-4-byte hash; a bucket probe whose hit-check is one SIMD key compare; then a single-`long` branchless number parse; update min/max/sum/count in the node. Threads run independent maps and merge into a sorted `TreeMap` at the end. The byte-at-a-time scan that was the bottleneck is gone, replaced by data-parallel compares that chew through 32 bytes per instruction — enough to roughly halve the wall-clock into the low-three-second range.

```java
private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED.length() >= 32
        ? ByteVector.SPECIES_256 : ByteVector.SPECIES_128;

// one iteration of the main loop: vectorized ';' search + inlined map probe
private static long iterate(PoorManMap aggrMap, MemorySegment data, long offset) {
    var line = ByteVector.fromMemorySegment(BYTE_SPECIES, data, offset, ByteOrder.nativeOrder());
    long semicolons = line.compare(VectorOperators.EQ, ';').toLong();   // SIMD: all lanes vs ';'

    if (semicolons == 0) {                       // name longer than the vector -> scalar fallback
        int keySize = BYTE_SPECIES.length();
        while (data.get(ValueLayout.JAVA_BYTE, offset + keySize) != ';') keySize++;
        var node = aggrMap.indexSimple(data, offset, keySize);
        return parseDataPoint(aggrMap, node, data, offset + 1 + keySize);
    }

    int keySize = Long.numberOfTrailingZeros(semicolons);              // bitmask -> key length, no loop
    int x, y;
    if (keySize >= Integer.BYTES) {
        x = data.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        y = data.get(ValueLayout.JAVA_INT_UNALIGNED, offset + keySize - Integer.BYTES);
    } else {
        x = data.get(ValueLayout.JAVA_BYTE, offset);
        y = data.get(ValueLayout.JAVA_BYTE, offset + keySize - Byte.BYTES);
    }
    int hash = PoorManMap.hash(x, y);
    int bucket = hash & PoorManMap.BUCKET_MASK;
    Aggregator node;
    for (;; bucket = (bucket + 1) & PoorManMap.BUCKET_MASK) {
        node = aggrMap.nodes[bucket];
        if (node == null) { node = aggrMap.insertInto(bucket, data, offset, keySize); break; }
        if (node.keySize != keySize) continue;
        var nodeKey = ByteVector.fromArray(BYTE_SPECIES, aggrMap.keyData, bucket * PoorManMap.KEY_SIZE);
        long eqMask = line.compare(VectorOperators.EQ, nodeKey).toLong();   // SIMD key compare
        long validMask = semicolons ^ (semicolons - 1);                    // bytes up to ';'
        if ((eqMask & validMask) == validMask) break;
    }
    return parseDataPoint(aggrMap, node, data, offset + keySize + 1);
}

static int hash(int x, int y) {                  // FxHash on first-4 / last-4 bytes
    int seed = 0x9E3779B9, rotate = 5;
    return (Integer.rotateLeft(x * seed, rotate) ^ y) * seed;
}

// branchless SWAR parse of D.D / DD.D / -D.D / -DD.D into temperature*10
private static long parseDataPoint(PoorManMap aggrMap, Aggregator node, MemorySegment data, long offset) {
    long word = data.get(JAVA_LONG_LT, offset);
    int decimalSepPos = Long.numberOfTrailingZeros(~word & 0x10101000);   // digit bit 4 set, '.' not
    int shift = 28 - decimalSepPos;
    long signed = (~word << 59) >> 63;                                    // -1 if '-', else 0
    long designMask = ~(signed & 0xFF);
    long digits = ((word & designMask) << shift) & 0x0F000F0F00L;
    long absValue = ((digits * 0x640a0001) >>> 32) & 0x3FF;               // magic multiply, value*10
    long value = (absValue ^ signed) - signed;                           // apply sign
    aggrMap.observe(node, value);
    return offset + (decimalSepPos >>> 3) + 3;
}
```
