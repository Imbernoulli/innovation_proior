The branchless-SWAR-over-Unsafe rung landed at **00:02.157**, and the optimization log shows it came from steady small cuts rather than one trick — which tells me the algorithm is essentially right: parallel, byte-keyed open addressing, scaled-integer aggregation, branchless register-width parsing over raw addresses. There is no obvious *new* algorithm left to find. What remains is two kinds of overhead the previous rung still pays, and to get under two seconds I attack both at once: the fixed cost the JVM imposes before and around the actual work, and the last branch-misprediction and load-stall slack inside the hot loop.

The method is the **GraalVM ahead-of-time native binary with work-stealing 2 MB segments and three interleaved cursors** (the thomaswue composition). Start with the fixed cost, because on a ~2-second job it is a real fraction of the total. The previous rung still runs on the JVM: it starts the VM, loads classes, and lets the JIT warm up by interpreting and then recompiling the hot loop while the clock is already running. For a long job that warmup amortizes to nothing, but here the whole run is two seconds, so the seconds of startup and the early interpreted iterations are measurable. The fix is ahead-of-time compilation with GraalVM `native-image` into a standalone binary: no VM to start, no class loading, no JIT warmup — the hot loop is already optimized machine code from the first instruction. The rules forbid baking the answer in at build time, so the native image still reads the file and does all the work at runtime; I remove only the *startup and warmup*. I keep the subprocess unmap trick exactly: `spawnWorker` re-launches the process with `--worker`, the worker does the work and prints, and the slow ~12 GB unmap happens after the answer is already out.

The first hot-loop refinement is **work stealing with small segments**. The previous rung split the file into one equal slice per core. The problem with an equal split is *imbalance*: page faults as the mmap warms, NUMA effects, and the luck of which thread hits more distinct keys mean some threads finish early and idle while a straggler runs long — and total time is the *slowest* thread, so idle cores at the end are wasted. Instead I carve the file into many small segments, `SEGMENT_SIZE = 1<<21 = 2 MB`, handed out from a single shared `AtomicLong` cursor that every thread `addAndGet`s. A thread that finishes a 2 MB segment immediately grabs the next; no thread is left holding a long tail because the granularity is fine and work flows to whichever core is free. This keeps all eight cores busy right to the last byte. The granularity is the tuning knob: too small and the `AtomicLong` becomes the bottleneck, too large and the imbalance returns; 2 MB is the credited sweet spot.

The second refinement is **three cursors in the same thread**. Each 2 MB segment is split into three parts, and I advance a `Scanner` over each, interleaving their work in the loop body — read `w1`, `w2`, `w3`; find three delimiters; do three lookups; parse three numbers; record three results. The reason this helps is instruction-level parallelism and memory-latency hiding: each line's processing is a dependent chain (load → find delimiter → hash → probe → load entry → update), and a single cursor stalls the core whenever a load misses cache. With three independent cursors in flight, while cursor 1 waits on a load the out-of-order engine has cursors 2 and 3's independent work to chew on, so the core stays fed. Three is the sweet spot — enough independent streams to hide latency, few enough to keep their working sets in registers and L1.

The per-line work is the same SWAR family, tuned to shave the last branches. The delimiter search is $\text{findDelimiter}(w) = (\text{input} - \mathtt{0x01..01})\ \&\ \lnot\text{input}\ \&\ \mathtt{0x80..80}$ after XOR with `0x3B..3B`. The credited insight here is to read the first **16 bytes** of every name *unconditionally* as two words (`word`, `wordB`) with their two delimiter masks, because branching between the "<8-byte" and "8–16-byte" cases costs more in misprediction than the wasted work of always doing both. So I compute `letterCount1` and `letterCount2` from the two masks, build the key from the two words with precomputed `MASK1`/`MASK2` lookup tables — mask-based rather than bit-shifting, since table lookups avoid variable shifts — and form the hash as `word ^ word2`. Only names longer than 16 bytes take the slow scalar-ish path that keeps scanning 8 bytes at a time. The number parse is the same branchless magic-constant SWAR (`convertIntoNumber`, credited to Quan Anh Mai): find the `'.'` with $\lnot\text{numberWord}\ \&\ \mathtt{0x10101000}$, recover the sign, align with a shift of $28 - \text{decimalSepPos}$ and multiply by `0x640a0001` to sum the digits into bits 32–41, mask `& 0x3FF`, apply sign — temperature times ten, no branches.

The map is open addressing again, sized `1<<17`, but the entry is now a real `Result` whose first two fields are the name's two `long` words (`firstNameWord`, `secondNameWord`) plus the raw `nameAddress`. On a probe the cheap check compares those two cached words directly (`existingResult.firstNameWord == word && existingResult.secondNameWord == word2`); only on a rare long-name collision do I walk the name 8 bytes at a time against the stored `nameAddress`, probing `(tableIndex + 31) & (len-1)` on collision. Aggregation keeps min/max as `short` and sum as `long` — the scaled-integer values fit a `short` exactly — and `record` updates them branch-lean. Everything reads through `Scanner`'s `Unsafe.getLong(pos)` on raw mmap addresses; the per-thread `Result[]` means no locks in the loop; a final `TreeMap` accumulation (`putIfAbsent` + `accumulate`) merges the per-thread results and sorts for output, and the name is decoded to a `String` only at the end, once per distinct station, in `calcName`.

So the finale is the same parallel byte-keyed scaled-integer SWAR algorithm brought to its limit on three axes simultaneously: an AOT GraalVM native binary that erases JVM startup and JIT warmup; a work-stealing scheduler over 2 MB segments from a shared atomic cursor that keeps all eight cores busy to the last byte; and a three-cursors-per-thread inner loop with unconditional 16-byte name reads and table-mask key building that hides memory latency and strips the last data-dependent branches. Each piece is one contributor's idea — the subprocess and Unsafe mmap (Alfonso Peterssen), the work-stealing 2 MB segments (Artsiom Korzun), the branch-free <8/8–16 unification (Jaromir Hamala), the mask-table lookups (Van Phu DO), the branchless number parse (Quan Anh Mai) — composed into one program. Removing the JVM's fixed startup/warmup tax, the idle-core tail of an equal split, and the per-line load stalls and length branches should compound into the final sub-two-second result, making this the fastest known way to aggregate a billion rows on this machine and the rank-1 entry at **00:01.535**.

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
