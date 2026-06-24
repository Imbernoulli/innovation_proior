import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    N = int(data[idx]); idx += 1
    w = []
    c = []
    for i in range(m):
        w.append(int(data[idx])); idx += 1
        c.append(int(data[idx])); idx += 1

    if N == 0:
        print(0)
        return

    def produced(T):
        total = 0
        for i in range(m):
            if T >= w[i]:
                total += (T - w[i]) // c[i] + 1
        return total

    # Independent method: simple linear scan over candidate times.
    # Only the times when SOME press emits a part can change produced(T),
    # but to be obviously correct we just scan every millisecond from 0 upward
    # until produced(T) >= N. Generator keeps the answer small.
    T = 0
    while produced(T) < N:
        T += 1
    print(T)

main()
