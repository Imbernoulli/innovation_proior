# TIER: greedy
# The obvious first idea: review the items the MODEL is most confident are
# violations first ("catch the most violations"), filling one reviewer to their
# capacity before moving to the next. This ignores severity, appeal cost, and
# ambiguity entirely, and never rotates reviewers deliberately. It reliably
# burns scarce reviewer capacity on the model's most CONFIDENT predictions --
# exactly the items the cheap auto-rule was already most likely to get right --
# while severe-but-uncertain items (moderate model_score, high severity) sit
# untouched and get auto-actioned.
import sys, json

inst = json.load(sys.stdin)
items = inst["items"]
reviewers = inst["reviewers"]

order = sorted(range(len(items)), key=lambda i: -items[i]["model_score"])

schedule = {}
idx = 0
for rv in reviewers:
    rid = str(rv["id"])
    cap = rv["capacity"]
    lst = []
    while idx < len(order) and len(lst) < cap:
        lst.append(order[idx])
        idx += 1
    schedule[rid] = lst

print(json.dumps({"schedule": schedule}))
