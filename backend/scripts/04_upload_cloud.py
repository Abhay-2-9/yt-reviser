import json, os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "yt_reviser"
MODEL = "sentence-transformers/all-minilm-l6-v2"   # embedded inside Qdrant Cloud
BATCH = 32

chunks = [json.loads(l) for l in open("data/knowledge.jsonl", encoding="utf-8")]
print(len(chunks), "chunks to upload")

client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                      cloud_inference=True)

if not client.collection_exists(COLLECTION):
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
client.create_payload_index(COLLECTION, "category", models.PayloadSchemaType.KEYWORD)

for i in range(0, len(chunks), BATCH):
    batch = chunks[i : i + BATCH]
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=c["id"],
                # Qdrant embeds this text itself (title + text, same as before)
                vector=models.Document(text=f"{c['title']}. {c['text']}", model=MODEL),
                payload={k: v for k, v in c.items() if k != "id"},
            )
            for c in batch
        ],
    )
    print(f"uploaded {min(i + BATCH, len(chunks))}/{len(chunks)}")

print("Points in Qdrant:", client.count(COLLECTION).count)