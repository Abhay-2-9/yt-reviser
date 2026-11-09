import json, uuid

MAX_WORDS = 80     # ~30-45 sec of speech; safely under MiniLM's 256-token limit
OVERLAP_SEGS = 1   # repeat the last segment so a topic isn't cut at a boundary

def chunk(segs):
    i = 0
    while i < len(segs):
        j, words = i, 0
        while j < len(segs) and words < MAX_WORDS:
            words += len(segs[j]["text"].split())
            j += 1
        yield segs[i:j]
        if j >= len(segs):
            break
        i = max(j - OVERLAP_SEGS, i + 1)

n_videos = n_chunks = 0
with open("data/knowledge.jsonl", "w", encoding="utf-8") as out:
    for line in open("data/transcripts.jsonl", encoding="utf-8"):
        v = json.loads(line)
        n_videos += 1
        base = f"https://www.youtube.com/watch?v={v['video_id']}"
        for n, c in enumerate(chunk(v["segments"])):
            start = int(c[0]["start"])
            rec = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{v['video_id']}_{n}")),
                "category": v["category"],
                "title": v["title"],
                "start": start,
                "end": int(c[-1]["end"]),
                "link_with_time": f"{base}&t={start}s",
                "text": " ".join(s["text"] for s in c)         # this is what we embed
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_chunks += 1

print(f"{n_videos} videos -> {n_chunks} chunks (avg {n_chunks // max(n_videos, 1)} per video)")