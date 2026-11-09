import json, os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()
COLLECTION = "yt_reviser"
BATCH = 64

chunks = [json.loads(l) for l in open("data/knowledge.jsonl", encoding="utf-8")]
print(len(chunks), "chunks")

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)   # 384 dims

# Qdrant Cloud if QDRANT_URL is set in .env
url = os.getenv("QDRANT_URL")
client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))

if not client.collection_exists(COLLECTION):
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

# Index for fast category filtering
client.create_payload_index(COLLECTION, "category", models.PayloadSchemaType.KEYWORD)

for i in range(0, len(chunks), BATCH):
    batch = chunks[i : i + BATCH]
    # embed title + text; timestamps are not embedded
    vecs = model.encode([f"{c['title']}. {c['text']}" for c in batch],
                        normalize_embeddings=True, show_progress_bar=False)
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=c["id"],
                vector=v.tolist(),
                payload={k: val for k, val in c.items() if k != "id"},
            )
            for c, v in zip(batch, vecs)
        ],
    )
    print(f"uploaded {min(i + BATCH, len(chunks))}/{len(chunks)}")