# TIER: greedy
# The textbook reflex: classic best-fit / lowest-address free-list search, and
# ALWAYS join a freed berth with its sibling right away (the "safe" hygiene
# habit that keeps the freelist tidy). It never looks at which sectors the
# crew's roster currently remembers, and it never asks whether the berth it
# just tidied away will be wanted again in a moment.
import sys


def main():
    it = sys.stdin.buffer.read().split()
    p = 0
    heap = int(it[p]); page = int(it[p + 1]); tlb = int(it[p + 2]); n = int(it[p + 3])
    p += 4

    free_blocks = {0: heap}
    alloc_map = {}
    out = []

    def find_ancestor(addr, size):
        for a, s in free_blocks.items():
            if s >= size and a <= addr < a + s and (addr - a) % size == 0:
                return a, s
        return None

    def choose_alloc(size):
        exact = [a for a, s in free_blocks.items() if s == size]
        if exact:
            return min(exact)
        bigger = [(s, a) for a, s in free_blocks.items() if s > size]
        if not bigger:
            bigger = [(s, a) for a, s in free_blocks.items() if s >= size]
        bigger.sort()
        min_s = bigger[0][0]
        same = [a for s, a in bigger if s == min_s]
        return min(same)

    def do_alloc(id_, size):
        addr = choose_alloc(size)
        a, s = find_ancestor(addr, size)
        del free_blocks[a]
        while s > size:
            s //= 2
            left, right = a, a + s
            if addr < right:
                keep, other = left, right
            else:
                keep, other = right, left
            free_blocks[other] = s
            a = keep
        alloc_map[id_] = (addr, size)
        out.append("A %d %d" % (id_, addr))

    def do_free(id_):
        addr, size = alloc_map.pop(id_)
        flag = 1  # always tidy up immediately
        out.append("F %d %d" % (id_, flag))
        free_blocks[addr] = size
        if flag == 1:
            a, s = addr, size
            while s < heap:
                buddy = a ^ s
                if free_blocks.get(buddy) == s:
                    del free_blocks[a]; del free_blocks[buddy]
                    a = min(a, buddy); s *= 2
                    free_blocks[a] = s
                else:
                    break

    for _ in range(n):
        typ = it[p].decode(); p += 1
        if typ == 'A':
            id_ = int(it[p]); size = int(it[p + 1]); p += 2
            do_alloc(id_, size)
        elif typ == 'F':
            id_ = int(it[p]); p += 1
            do_free(id_)
        else:  # 'T'
            p += 2

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
