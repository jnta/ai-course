import os
import uuid

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from utils.semantic_chunker import SemanticChunker

load_dotenv()

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
FILE_PATH = "./projeto/financial_file.md"
MAX_TOKENS = 300

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))

if qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(
        f"Coleção {COLLECTION_NAME} já existe. Removendo para aplicar nova configuração..."
    )
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

print(f"Criando coleção {COLLECTION_NAME} com suporte a Dense, Sparse e ColBERT...")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={
        "sparse_vector": models.SparseVectorParams(),
    },
)

with open(FILE_PATH, "r") as f:
    content = f.read()


chunks = SemanticChunker(max_tokens=MAX_TOKENS).create_chunks(content)

dense_model = TextEmbedding(DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_NAME)

print(f"Modelos carregados. Processando {len(chunks)} chunks...")

points = []
for chunk in chunks:
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
        payload={"text": chunk, "source": "financial_file.md"},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)
print("Ingestão concluída com sucesso!")
