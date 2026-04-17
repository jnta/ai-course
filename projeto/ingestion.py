import os
import uuid

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from utils.edgar_client import EdgarClient
from utils.semantic_chunker import SemanticChunker

load_dotenv()

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
MAX_TOKENS = 300

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))

edgar_client = EdgarClient(os.getenv("EDGAR_EMAIL"))

data_10k = edgar_client.fetch_flining_data("AAPL", "10-K")
data_10q = edgar_client.fetch_flining_data("AAPL", "10-Q")

content_10k = edgar_client.get_combined_text(data_10k)
content_10q = edgar_client.get_combined_text(data_10q)

chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
for data, text in [(data_10k, content_10k), (data_10q, content_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": data["metadata"]})

dense_model = TextEmbedding(DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)

print(f"Modelos carregados. Processando {len(all_chunks)} chunks...")

points = []
for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_emb = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_emb = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_emb = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_emb,
            "sparse_vector": sparse_emb,
            "colbert": colbert_emb,
        },
        payload={"text": chunk, "metadata": metadata},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=5)
print("Ingestão concluída com sucesso!")
