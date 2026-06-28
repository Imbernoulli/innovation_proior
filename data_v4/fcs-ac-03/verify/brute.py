import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    q = int(data[idx]); idx += 1
    nums = []          # inserted numbers (in order)
    out_lines = []

    def all_xor_values():
        # Set of all subset-XOR values of the current collection (empty subset -> 0).
        vals = {0}
        for x in nums:
            new = set()
            for v in vals:
                new.add(v ^ x)
            vals |= new
        return vals

    for _ in range(q):
        t = int(data[idx]); idx += 1
        if t == 1:
            x = int(data[idx]); idx += 1
            nums.append(x)
        elif t == 2:
            vals = all_xor_values()
            out_lines.append(str(max(vals)))
        elif t == 3:
            x = int(data[idx]); idx += 1
            vals = all_xor_values()
            out_lines.append("YES" if x in vals else "NO")
        elif t == 4:
            k = int(data[idx]); idx += 1
            vals = sorted(all_xor_values())   # distinct, ascending
            if 1 <= k <= len(vals):
                out_lines.append(str(vals[k - 1]))
            else:
                out_lines.append("-1")
        else:
            raise ValueError("bad type")

    sys.stdout.write("\n".join(out_lines))
    if out_lines:
        sys.stdout.write("\n")

main()
