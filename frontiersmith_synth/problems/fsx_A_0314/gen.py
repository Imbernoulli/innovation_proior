import sys

def main():
    i = int(sys.argv[1])
    # difficulty ladder: number of turbines grows 12 -> 21.
    # More turbines on the same field -> the thinnest achievable wake triangle
    # shrinks and the layout problem gets combinatorially harder.
    n = 11 + i          # i = 1..10  ->  n = 12..21
    if n < 3:
        n = 3
    out = []
    out.append(str(n))
    out.append("0 1 0 1")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
