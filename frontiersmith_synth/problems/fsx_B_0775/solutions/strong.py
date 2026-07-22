# TIER: strong
# Genuine insight: don't trust a single canonical multiplicative constant, and don't
# reduce via mod-M before mixing. Read the ACTUAL published sweep, then search among
# multiply+top-bits candidates for one whose per-family peak load is small for THIS
# specific sweep -- i.e. characterize the sweep's planted resonances (empirically, by
# direct evaluation) and pick a mixer whose blind spot does not overlap them, rather than
# applying one fixed textbook trick and hoping it averages out.
import sys

MASK64 = (1 << 64) - 1
M = 1024
SHIFT = 54


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    f = int(next(it))
    fams = []
    for _ in range(f):
        kind = next(it)
        if kind == "AP":
            start = int(next(it)); stride = int(next(it)); count = int(next(it))
            fams.append([(start + i * stride) & MASK64 for i in range(count)])
        elif kind == "COSET":
            base = int(next(it)); lo = int(next(it)); width = int(next(it))
            winmask = ((1 << width) - 1) << lo
            base = base & (~winmask & MASK64)
            fams.append([base | (pattern << lo) for pattern in range(1 << width)])
        elif kind == "FLOAT":
            exp = int(next(it)); mb = int(next(it)); ms = int(next(it)); count = int(next(it))
            mant_mask = (1 << 40) - 1
            fams.append([((exp << 40) | ((mb + i * ms) & mant_mask)) & MASK64
                         for i in range(count)])
    return m, fams


def peak_of_peaks(fams, a):
    worst = 0
    for keys in fams:
        buckets = [0] * M
        for x in keys:
            b = ((a * x) & MASK64) >> SHIFT
            buckets[b] += 1
            if buckets[b] > worst:
                worst = buckets[b]
    return worst


class Lcg:
    """Small deterministic PRNG (fixed seed) so the search is reproducible without
    relying on the platform's `random` module seeding behavior."""
    def __init__(self, seed):
        self.x = seed & MASK64

    def next64(self):
        self.x = (6364136223846793005 * self.x + 1442695040888963407) & MASK64
        return self.x


def main():
    m, fams = read_instance()
    total_keys = sum(len(k) for k in fams)
    # keep runtime comfortably under the time limit for the largest sweeps
    trials = max(40, min(300, 3_000_000 // max(1, total_keys)))

    rng = Lcg(0x123456789ABCDEF0)
    best_val = None
    best_a = None
    for _ in range(trials):
        a = rng.next64() | 1
        val = peak_of_peaks(fams, a)
        if best_val is None or val < best_val:
            best_val = val
            best_a = a

    print("1 TOPBITS")
    print(f"MUL {best_a} 0")


if __name__ == "__main__":
    main()
