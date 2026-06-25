import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]; idx += n
    b = [int(data[idx + i]) for i in range(m)]; idx += m

    # Brute force: enumerate all n*m products, sort, take the K-th smallest (1-indexed).
    prods = []
    for x in a:
        for y in b:
            prods.append(x * y)
    prods.sort()
    print(prods[K - 1])

if __name__ == "__main__":
    main()
