# TIER: trivial
# Independent binary (double-and-add) chain per target, concatenated. Matches the
# checker's baseline B exactly -> score ~0.1.
import sys

def main():
    d = sys.stdin.read().split()
    k = int(d[0]); targets = [int(x) for x in d[1:1 + k]]
    seq = [1]
    for t in targets:
        bits = bin(t)[2:]
        cur = 1
        for ch in bits[1:]:
            cur = cur * 2          # = prev + prev
            seq.append(cur)
            if ch == '1':
                cur = cur + 1      # = cur + 1 (1 is present)
                seq.append(cur)
    sys.stdout.write(' '.join(map(str, seq)) + '\n')

if __name__ == "__main__":
    main()
