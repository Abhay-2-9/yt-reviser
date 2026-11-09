import csv, os
from yt_dlp import YoutubeDL

PLAYLIST_URL = "https://youtube.com/playlist?list=PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2&si=Wlpd9sMlmAEhrnaq"

os.makedirs("data", exist_ok=True)
with open("data/videos.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["url", "title", "category"])
    with YoutubeDL({"quiet": True, "extract_flat": True}) as y:
        info = y.extract_info(PLAYLIST_URL, download=False)
        count = 0
        for e in info["entries"]:
            if not e:                      # skip deleted/private videos
                continue
            w.writerow([f"https://www.youtube.com/watch?v={e['id']}", e["title"], ""])
            count += 1

print(f"Saved {count} videos to data/videos.csv")