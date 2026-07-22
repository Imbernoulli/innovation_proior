# TIER: invalid
# Plausible-looking but wrong: claims to transcribe every line, but injects
# a bogus extra token so the rewrite never expands back to the original --
# must be rejected (score 0) on every instance.
import sys, json


def main():
    inst = json.load(sys.stdin)
    lines = inst["lines"]
    rewrites = [list(line) + ["__BOGUS__"] for line in lines]
    print(json.dumps({"new_macros": [], "rewrites": rewrites}))


main()
