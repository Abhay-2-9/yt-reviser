import json, re, sys

# Order matters: the first match wins. "Linked List Slow & Fast Pointer" must hit
# Linked List before Two Pointers, so Linked List comes first.
RULES = [
    ("Linked List",         [r"linked list", r"slow and fast pointer", r"slow & fast pointer", r"linklist"]),
    ("Sliding Window",      [r"sliding window"]),
    ("Two Pointers",        [r"two pointer", r"2 pointer", r"two-pointer"]),
    ("Binary Search",       [r"binary search", r"koko"]),
    ("Prefix Sum",          [r"prefix sum"]),
    ("Time Complexity",     [r"time complexity"]),
    ("Dynamic Programming", [r"\bdp\b", r"dynamic programming", r"knap", r"subsequence",
                             r"\blcs\b", r"\blis\b", r"tabulation", r"memoization"]),
    ("Greedy",              [r"greedy"]),
    ("Bit Manipulation",    [r"bit manipulation", r"\bxor\b"]),
    ("Stack & Queue",       [r"\bstack\b", r"\bqueue\b", r"monotonic"]),
    ("Recursion & Backtracking", [r"recursion", r"backtrack"]),
    ("Trees",               [r"\btree\b", r"\bbst\b"]),
    ("Graphs",              [r"graph", r"\bbfs\b", r"\bdfs\b", r"dijkstra"]),
    ("Heaps",               [r"\bheap\b", r"priority queue"]),
    ("Hashing",             [r"hash", r"hash table", r"hashmap"]),
    ("Strings",             [r"\bstring"]),
    ("Kadane's Algorithm",  [r"kadane"]),
    ("Merge Intervals",     [r"merge interval"]),
    ("Path Sum",            [r"path sum"]),
    ("Bellman Ford",        [r"bellman-ford"]),
    ("Topological Sort",    [r"topological sort"]),
]

def categorize(title):
    t = title.lower()
    for cat, patterns in RULES:
        if any(re.search(p, t) for p in patterns):
            return cat
    return "Other"

videos = [json.loads(l) for l in open("data/transcripts.jsonl", encoding="utf-8")]
for v in videos:
    v["category"] = categorize(v["title"])

for v in sorted(videos, key=lambda x: x["category"]):
    print(f"{v['category']:26} | {v['title']}")

counts = {}
for v in videos:
    counts[v["category"]] = counts.get(v["category"], 0) + 1
print("\n", counts)

if "--save" in sys.argv:
    with open("data/transcripts.jsonl", "w", encoding="utf-8") as f:
        for v in videos:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print("\nSaved categories to transcripts.jsonl")
else:
    print("\nDry run. Add --save to write.")