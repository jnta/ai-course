import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))
model = TextEmbedding(MODEL_NAME)

query_text = "what are the ain financial risks?"
query_embedding = list(model.passage_embed(query_text))[0].tolist()

result = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=models.Document(text=query_text, model=MODEL_NAME),
    limit=5,
)

for point in result.points:
    print(f"Score: {point.score:.4f}")
    print(f"Text: {point.payload['text']}")
    print("-" * 80)
