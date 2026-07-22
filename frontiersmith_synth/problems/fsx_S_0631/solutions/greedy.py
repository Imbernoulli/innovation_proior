# TIER: greedy
"""The obvious textbook recipe: splay every requested tag straight to the root
before its own search (classic move-to-root heuristic), regardless of season
weight or whether anyone will ever ask for that tag again.  Simulates its own
local copy of the tree (identical semantics to the evaluator) to know how many
rotate_up(key) calls are needed -- exactly `depth(key)`, since each call always
strictly reduces the target key's depth by one, whatever the tree shape."""
import sys, json


def build_tree(inst):
    n = inst["n"]
    leftArr = inst["left"]; rightArr = inst["right"]
    parent = [0] * (n + 1)
    left = [0] * (n + 1); right = [0] * (n + 1)
    for k in range(1, n + 1):
        l = leftArr[k - 1]; r = rightArr[k - 1]
        left[k] = l; right[k] = r
        if l:
            parent[l] = k
        if r:
            parent[r] = k
    return {"left": left, "right": right, "parent": parent, "root": inst["root"]}


def rotate_up(tree, k):
    par = tree["parent"]; left = tree["left"]; right = tree["right"]
    p = par[k]
    if p == 0:
        return False
    g = par[p]
    if left[p] == k:
        b = right[k]
        left[p] = b
        if b:
            par[b] = p
        right[k] = p
    else:
        b = left[k]
        right[p] = b
        if b:
            par[b] = p
        left[k] = p
    par[p] = k
    par[k] = g
    if g == 0:
        tree["root"] = k
    else:
        if left[g] == p:
            left[g] = k
        else:
            right[g] = k
    return True


def main():
    inst = json.load(sys.stdin)
    tree = build_tree(inst)
    ops = []
    for key in inst["accesses"]:
        oi = []
        while tree["parent"][key] != 0:
            oi.append(key)
            rotate_up(tree, key)
        ops.append(oi)
    print(json.dumps({"ops": ops}))


if __name__ == "__main__":
    main()
