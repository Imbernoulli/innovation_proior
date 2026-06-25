import sys

MOD = 1000000007

def main():
    data = sys.stdin.read().split()
    if not data:
        # no input at all
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    t = []
    for _ in range(n):
        t.append(int(data[idx])); idx += 1

    # Independent brute force: enumerate every one of the 2^n index subsets,
    # build the resulting tone-id sequence (in order), collect DISTINCT sequences
    # in a set, exclude the empty one, and count.
    distinct = set()
    for mask in range(1 << n):
        seq = []
        for i in range(n):
            if mask & (1 << i):
                seq.append(t[i])
        if seq:  # non-empty only
            distinct.add(tuple(seq))
    print(len(distinct) % MOD)

if __name__ == "__main__":
    main()
