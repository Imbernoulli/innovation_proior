**Problem (from step 4).** The branchless-SWAR rung (02.157 s) got the algorithm essentially right —
there's no new algorithm left. What remains is overhead: the JVM's fixed startup + JIT-warmup tax, which
is a real fraction of a ~2-second run; the idle-core tail of an equal file split (total time = slowest
thread); and the last per-line load stalls and data-dependent length branches.

**Key idea.** Bring the same parallel byte-keyed scaled-integer SWAR algorithm to its limit on three axes
at once. (1) Build with GraalVM `native-image` into an **ahead-of-time native binary** — no VM startup, no
class loading, no JIT warmup, computation still at runtime. (2) Replace the equal split with **work
stealing over 2 MB segments** handed out from a shared `AtomicLong` cursor, so every core stays busy to the
last byte. (3) Run **three cursors per thread** interleaved, so out-of-order execution hides each cursor's
memory-load stalls behind the others' independent work. Read the first 16 bytes of every name
*unconditionally* as two words (no <8 vs 8–16 branch), build the key and hash with `MASK1`/`MASK2` lookup
tables, and parse the number with the branchless magic-constant SWAR. Keep the subprocess unmap trick.
Everything reads raw mmap addresses via `Unsafe`.

**Why it works.** AOT compilation erases the startup/warmup tax that only matters because the job is short.
Fine-grained work stealing removes the straggler tail of a fixed split. Three independent cursors keep the
core fed while any one stalls on a cache miss (instruction-level parallelism / latency hiding). Doing both
name-length cases unconditionally trades a little wasted work for removing a branch that mispredicts a
billion times; mask-table key building avoids variable shifts. Each piece is a composed contributor idea
(Unsafe mmap + subprocess; 2 MB work stealing; branch-free <8/8–16 unification; mask-table lookups;
branchless number parse). This is the rank-1 result.

**Change / code.** The three-cursor work-stealing parse loop, the unconditional 16-byte name read with
mask-table key building, and the branchless SWAR delimiter/number primitives.

```java
private static final int SEGMENT_SIZE = 1 << 21;      // 2 MB work-stealing unit
private static final int HASH_TABLE_SIZE = 1 << 17;

private static void parseLoop(AtomicLong counter, long fileEnd, long fileStart, List<Result> collectedResults) {
    Result[] results = new Result[HASH_TABLE_SIZE];
    while (true) {
        long current = counter.addAndGet(SEGMENT_SIZE) - SEGMENT_SIZE;     // steal next 2 MB
        if (current >= fileEnd) return;
        long segmentEnd = nextNewLine(Math.min(fileEnd - 1, current + SEGMENT_SIZE));
        long segmentStart = (current == fileStart) ? current : nextNewLine(current) + 1;

        long dist = (segmentEnd - segmentStart) / 3;                       // three cursors per thread
        long midPoint1 = nextNewLine(segmentStart + dist);
        long midPoint2 = nextNewLine(segmentStart + dist + dist);
        Scanner s1 = new Scanner(segmentStart, midPoint1);
        Scanner s2 = new Scanner(midPoint1 + 1, midPoint2);
        Scanner s3 = new Scanner(midPoint2 + 1, segmentEnd);
        while (s1.hasNext() && s2.hasNext() && s3.hasNext()) {
            long w1 = s1.getLong(), w2 = s2.getLong(), w3 = s3.getLong();
            long d1 = findDelimiter(w1), d2 = findDelimiter(w2), d3 = findDelimiter(w3);
            long w1b = s1.getLongAt(s1.pos() + 8), w2b = s2.getLongAt(s2.pos() + 8), w3b = s3.getLongAt(s3.pos() + 8);
            long d1b = findDelimiter(w1b), d2b = findDelimiter(w2b), d3b = findDelimiter(w3b);
            Result r1 = findResult(w1, d1, w1b, d1b, s1, results, collectedResults);
            Result r2 = findResult(w2, d2, w2b, d2b, s2, results, collectedResults);
            Result r3 = findResult(w3, d3, w3b, d3b, s3, results, collectedResults);
            record(r1, scanNumber(s1)); record(r2, scanNumber(s2)); record(r3, scanNumber(s3));
        }
        // ... tails of s1/s2/s3 handled the same way, one cursor at a time
    }
}

private static final long[] MASK1 = { 0xFFL, 0xFFFFL, 0xFFFFFFL, 0xFFFFFFFFL, 0xFFFFFFFFFFL,
        0xFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL };
private static final long[] MASK2 = { 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0xFFFFFFFFFFFFFFFFL };

// read the first 16 bytes of the name unconditionally; no <8 vs 8-16 branch
private static Result findResult(long initialWord, long initialDelimiterMask, long wordB, long delimiterMaskB,
                                 Scanner scanner, Result[] results, List<Result> collectedResults) {
    long word = initialWord, delimiterMask = initialDelimiterMask, word2 = wordB, delimiterMask2 = delimiterMaskB;
    long hash; long nameAddress = scanner.pos();
    if ((delimiterMask | delimiterMask2) != 0) {
        int letterCount1 = Long.numberOfTrailingZeros(delimiterMask) >>> 3;   // 1..8
        int letterCount2 = Long.numberOfTrailingZeros(delimiterMask2) >>> 3;  // 0..8
        long mask = MASK2[letterCount1];
        word  = word  & MASK1[letterCount1];
        word2 = mask & word2 & MASK1[letterCount2];
        hash = word ^ word2;
        Result existingResult = results[hashToIndex(hash, results)];
        scanner.add(letterCount1 + (letterCount2 & mask));
        if (existingResult != null && existingResult.firstNameWord == word && existingResult.secondNameWord == word2)
            return existingResult;                                            // cached-word fast hit
    } else {
        hash = word ^ word2; scanner.add(16);                                 // name > 16 bytes: keep scanning
        while (true) {
            word = scanner.getLong(); delimiterMask = findDelimiter(word);
            if (delimiterMask != 0) { int tz = Long.numberOfTrailingZeros(delimiterMask);
                word = (word << (63 - tz)); scanner.add(tz >>> 3); hash ^= word; break; }
            else { scanner.add(8); hash ^= word; }
        }
    }
    // ... resolve table index, probe (tableIndex + 31) & (len-1) on collision, newEntry on miss
    return /* existingResult */ resolve(results, nameAddress, hash, scanner, collectedResults);
}

private static long findDelimiter(long word) {              // SWAR ';' search
    long input = word ^ 0x3B3B3B3B3B3B3B3BL;
    return (input - 0x0101010101010101L) & ~input & 0x8080808080808080L;
}

private static long scanNumber(Scanner scanPtr) {           // branchless SWAR parse (Quan Anh Mai)
    long numberWord = scanPtr.getLongAt(scanPtr.pos() + 1);
    int decimalSepPos = Long.numberOfTrailingZeros(~numberWord & 0x10101000L);
    long number = convertIntoNumber(decimalSepPos, numberWord);
    scanPtr.add((decimalSepPos >>> 3) + 4);
    return number;
}

private static long convertIntoNumber(int decimalSepPos, long numberWord) {
    int shift = 28 - decimalSepPos;
    long signed = (~numberWord << 59) >> 63;
    long designMask = ~(signed & 0xFF);
    long digits = ((numberWord & designMask) << shift) & 0x0F000F0F00L;
    long absValue = ((digits * 0x640a0001) >>> 32) & 0x3FF;
    return (absValue ^ signed) - signed;
}
```

**Target.** Beat 00:02.157 — into sub-two-second territory — by composing AOT native compilation
(no startup/warmup), 2 MB work stealing (no idle-core tail), and three interleaved cursors with branchless
16-byte key handling (hide load stalls, drop length branches). This is the rank-1 result.
