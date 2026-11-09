import os, sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
COLLECTION = "yt_reviser"

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)   # 384 dims
url = os.getenv("QDRANT_URL")
client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))

def fmt(sec):
    return f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"

def search(question, k=5):
    q = model.encode(question, normalize_embeddings=True).tolist()
    hits = client.query_points(COLLECTION, query=q, limit=k).points
    return [(h.score, h.payload) for h in hits]

def build_context(results):
    # this text goes into the Groq prompt later
    return "\n\n".join(
        f"Video: {p['title']}\nTime: {fmt(p['start'])}\nLink: {p['link_with_time']}\nTranscript: {p['text']}"
        for _, p in results
    )

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "koko eating bananas"
    for score, p in search(question):
        print(f"{score:.3f} | {p['title']} | {fmt(p['start'])}")
        print(f"      {p['link_with_time']}")
        print(f"      {p['text'][:150]}...\n")