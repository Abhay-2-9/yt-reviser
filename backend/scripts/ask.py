import os, sys
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
COLLECTION = "yt_reviser"

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)   # 384 dims
url = os.getenv("QDRANT_URL")
client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def fmt(sec):
    return f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"

def search(question, k=5):
    q = model.encode(question, normalize_embeddings=True).tolist()
    hits = client.query_points(COLLECTION, query=q, limit=k).points
    return [(h.score, h.payload) for h in hits]

def build_context(results):
    return "\n\n".join(
        f"Video: {p['title']}\nTime: {fmt(p['start'])}\nLink: {p['link_with_time']}\nTranscript: {p['text']}"
        for _, p in results
    )

SYSTEM = """You help a student find which video in a DSA playlist explains a topic.
Use ONLY the excerpts provided. For the best matching excerpt(s), give:
- the video title
- the timestamp
- the link
- one line on what is explained there
Copy titles, times and links exactly from the excerpts. If no excerpt fits the question,
say you could not find it in the playlist. Never invent a video, time or link."""

def ask(question):
    results = search(question)
    context = build_context(results)
    resp = groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ],
        temperature=0.2,
    )
    return results, resp.choices[0].message.content

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "ransom note problem"
    results, answer = ask(question)
    print("--- retrieved ---")
    for score, p in results:
        print(f"{score:.3f} | {p['title']} | {fmt(p['start'])}")
    print("\n--- answer ---")
    print(answer)