import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"
FILE_PATH = "./projeto/financial_file.md"

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))

if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=qdrant.get_embedding_size(MODEL_NAME),
            distance=models.Distance.COSINE,
        ),
    )

with open(FILE_PATH, "r") as f:
    content = f.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

model = TextEmbedding(MODEL_NAME)
points = []

for chunk in chunks:
    embedding = list(model.passage_embed(chunk))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": chunk, "source": "financial_file.md"},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)
