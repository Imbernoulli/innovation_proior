# TIER: strong
# Top-down hierarchical, reversal-aware factoring.  At each recursive call on
# a substring s, find the LARGEST "mirror core" inside s -- i.e. the widest
# even-length palindromic span X + reverse(X) centered at some gap (found by
# expand-around-center, the same primitive Manacher's algorithm uses).  Build
# X once, reuse it for the mirrored half with a single reverse-rule, and
# recurse on the (shorter, hard, unstructured) prefix/suffix around it.  This
# peels the planted nesting one level at a time from the outside in, which a
# bottom-up pair-frequency pass has no way to do: it never asks "does this
# span equal the reverse of that span", it only ever asks "which adjacent
# pair repeats most often".
import sys

sys.setrecursionlimit(100000)


def find_best_center(s):
    n = len(s)
    best_c, best_r = -1, 0
    for c in range(1, n):
        r = 0
        while c - r - 1 >= 0 and c + r < n and s[c - r - 1] == s[c + r]:
            r += 1
        if r > best_r:
            best_r, best_c = r, c
    return best_c, best_r


def main():
    t = sys.stdin.readline().rstrip("\n")

    rules = []
    cache = {}

    def new_rule(tag, *args):
        rules.append((tag,) + args)
        return len(rules)

    def compress(s):
        if s in cache:
            return cache[s]
        if len(s) == 1:
            rid = new_rule("T", s)
            cache[s] = rid
            return rid
        rs = s[::-1]
        if rs in cache and rs != s:
            rid = new_rule("R", cache[rs])
            cache[s] = rid
            return rid

        c, r = find_best_center(s)
        if r >= 1:
            prefix, core, suffix = s[:c - r], s[c - r:c], s[c + r:]
            x_id = compress(core)
            rev_str = core[::-1]
            rev_id = cache.get(rev_str)
            if rev_id is None:
                rev_id = new_rule("R", x_id)
                cache[rev_str] = rev_id
            ids = []
            if prefix:
                ids.append(compress(prefix))
            ids.append(x_id)
            ids.append(rev_id)
            if suffix:
                ids.append(compress(suffix))
            cur = ids[0]
            for nxt in ids[1:]:
                cur = new_rule("C", cur, nxt)
            cache[s] = cur
            return cur

        mid = len(s) // 2
        left_id = compress(s[:mid])
        right_id = compress(s[mid:])
        rid = new_rule("C", left_id, right_id)
        cache[s] = rid
        return rid

    compress(t)

    out = []
    for r in rules:
        if r[0] == "T":
            out.append("T %s" % r[1])
        elif r[0] == "C":
            out.append("C %d %d" % (r[1], r[2]))
        else:
            out.append("R %d" % r[1])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
