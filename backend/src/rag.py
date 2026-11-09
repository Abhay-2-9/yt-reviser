import json, os, re
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "yt_reviser"

MODEL = "sentence-transformers/all-minilm-l6-v2"   # embedded inside Qdrant Cloud, not on this server
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                      cloud_inference=True)
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM = """You help a student find which video in a DSA playlist explains a topic.
You get numbered excerpts from videos. Decide which excerpts really explain the student's topic.
Reply with ONLY a JSON object, no markdown: {"ranking": [excerpt numbers that truly answer the question, best first, at most 3]}
If no excerpt explains the topic, reply {"ranking": []}. Use ONLY the excerpts."""


def fmt(sec):
    return f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"


def video_id(link):
    m = re.search(r"v=([\w-]{11})", link)
    return m.group(1) if m else None


@lru_cache(maxsize=1)          # restart the server after uploading new videos
def list_categories():
    cats, offset = set(), None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=500, offset=offset,
                                    with_payload=["category"], with_vectors=False)
        cats.update(p.payload["category"] for p in pts if p.payload.get("category"))
        if offset is None:
            break
    return sorted(cats)


def search(question, k=5, category=None):
    flt = None
    if category:
        flt = models.Filter(must=[
            models.FieldCondition(key="category", match=models.MatchValue(value=category))
        ])
    hits = client.query_points(COLLECTION, query=models.Document(text=question, model=MODEL), query_filter=flt, limit=k).points
    return [(h.score, h.payload) for h in hits]


def build_context(results):
    return "\n\n".join(
        f"Excerpt {i}\nVideo: {p['title']}\nTime: {fmt(p['start'])}\nTranscript: {p['text']}"
        for i, (_, p) in enumerate(results, 1)
    )


def answer_question(question, k=5, category=None):
    results = search(question, k, category)
    if not results:
        return {"answer": "I couldn't find anything for that.", "sources": []}

    answer = None
    try:
        resp = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{build_context(results)}"},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        ranking = [int(i) for i in data.get("ranking", [])]
    except Exception as e:                       # fall back to the search order
        print("Groq/parse error:", e)
        ranking = list(range(1, min(3, len(results)) + 1))

    order, seen = [], set()
    for i in ranking:                            # Groq's order: best video first
        if 1 <= i <= len(results) and i not in seen:
            seen.add(i); order.append(i)
    order = order[:3]

    if not order:                                # Groq found nothing relevant: show no videos
        return {"answer": answer or "I couldn't find that in the playlist.", "sources": []}

    sources = []
    for rank, i in enumerate(order, 1):
        score, p = results[i - 1]
        vid = video_id(p["link_with_time"])
        sources.append({
            "rank": rank,
            "title": p["title"],
            "category": p["category"],
            "timestamp": fmt(p["start"]),
            "link": p["link_with_time"],
            "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else None,
            "score": round(score, 3),
            "text": p["text"],
        })
    return {"answer": "\n".join(f"{s['title']} — {s['timestamp']}" for s in sources), "sources": sources}