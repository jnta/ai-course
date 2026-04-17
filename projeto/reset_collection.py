import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "financial"

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
