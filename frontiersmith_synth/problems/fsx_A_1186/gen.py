import sys
from _model import build


def main():
    test_id = int(sys.argv[1])
    inst = build(test_id)
    n, m = inst['n'], inst['m']
    out = []
    out.append("%d %d %d %d %d" % (n, m, inst['K'], inst['L'], inst['seed']))
    out.append(str(len(inst['observed'])))
    for (i, j, v) in inst['observed']:
        out.append("%d %d %.2f" % (i, j, v))
    out.append(str(len(inst['row_edges'])))
    for (a, b) in inst['row_edges']:
        out.append("%d %d" % (a, b))
    out.append(str(len(inst['col_edges'])))
    for (a, b) in inst['col_edges']:
        out.append("%d %d" % (a, b))
    out.append(str(len(inst['query'])))
    for (i, j) in inst['query']:
        out.append("%d %d" % (i, j))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
