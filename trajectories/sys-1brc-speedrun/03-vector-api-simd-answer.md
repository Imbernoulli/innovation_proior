**Problem (from step 2).** After parallel mmap (05.979 s), the per-row overhead and idle cores are gone;
what now dominates each core is the **scan** — the previous loop walks every name one byte at a time
testing each against `';'`, a billion times, with branchy, variable-length, bounds-checked reads. Finding
the field boundary and comparing keys byte-by-byte is the new hot spot.

**Key idea.** Use explicit SIMD via the JDK Vector API. Load the name region as a `ByteVector`
(`SPECIES_256`, 32 bytes) from a `MemorySegment` and find `';'` with one `line.compare(EQ, ';').toLong()`
— the bitmask's trailing-zero count *is* the key length, no loop. Hash from a fixed first-4 / last-4 bytes
with an FxHash step (constant work, any length). On a probe hit, compare keys with a single SIMD compare,
validating only the bytes before the delimiter via `validMask = semicolons ^ (semicolons - 1)`. Parse the
temperature with a branchless SWAR trick on one `long`. Long names and the segment tail fall back to
scalar slow paths.

**Why it works.** The byte-at-a-time delimiter search and key compare are replaced by data-parallel
compares that process 32 bytes per instruction, and the hash is fixed-cost regardless of name length, so
the per-line branch count and instruction count drop sharply. The 128K-bucket open-addressing table keeps
the load factor tiny (≤10K keys) so probes are short; per-thread maps keep the hot loop lock-free; the
SWAR number parse is a dozen branchless integer ops. The scan that was the bottleneck after parallelizing
is no longer linear in the name length on the common path.

**Change / code.** The vectorized line iteration (delimiter search + inlined hash-table probe + SIMD key
compare) and the branchless SWAR number parse.

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

**Target.** Beat 00:05.979 — roughly halve it, into the low-three-second range — by replacing the
byte-at-a-time delimiter scan and key compare with explicit SIMD compares.
