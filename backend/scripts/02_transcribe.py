import os, sys, site, glob

# Windows: make the pip-installed CUDA DLLs findable
for sp in site.getsitepackages() + [os.path.join(sys.prefix, "Lib", "site-packages")]:
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

import csv, json, re, tempfile, time
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

OUT = "data/transcripts.jsonl"

# TEST MODE: START = which row to begin at, LIMIT = how many videos
# Full run later: START = 0, LIMIT = None
START, LIMIT = 0, None

# CPU: ("medium", "cpu", "int8")   NVIDIA GPU: ("large-v3", "cuda", "float16")
MODEL, DEVICE, COMPUTE = "large-v3", "cuda", "int8_float16"

# Helps Whisper spell your topic words correctly
PROMPT = ("A DSA lecture on patterns: two pointers, sliding window, linked list, "
          "slow and fast pointer, binary search, Koko eating bananas, dynamic programming, "
          "knapsack, LIS, LCS, greedy, bit manipulation, XOR, prefix sum.")

model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["video_id"] for l in open(OUT, encoding="utf-8")}

def transcribe(url):
    with tempfile.TemporaryDirectory() as d:
        with YoutubeDL({"format": "bestaudio/best", "outtmpl": f"{d}/a.%(ext)s", "quiet": True}) as y:
            y.download([url])
        path = glob.glob(f"{d}/a.*")[0]
        segs, info = model.transcribe(
            path,
            task="translate",                  # Hindi/Hinglish -> English
            vad_filter=True,                   # skips silence
            beam_size=5,
            condition_on_previous_text=False,  # avoids repetition loops
            initial_prompt=PROMPT,
        )
        out = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()}
               for s in segs]
        return out, info.duration

with open("data/videos.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows = rows[START : (START + LIMIT) if LIMIT else None]

with open(OUT, "a", encoding="utf-8") as out:
    for row in rows:
        vid = re.search(r"v=([\w-]{11})", row["url"]).group(1)
        if vid in done:
            print("SKIP (already done)", row["title"])
            continue
        try:
            t0 = time.time()
            segments, duration = transcribe(row["url"])
            rec = {"video_id": vid, "title": row["title"], "link": row["url"],
                   "category": row["category"], "segments": segments}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"OK  {row['title']}\n    video {duration/60:.1f} min, took {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print("FAILED", row["url"], e)