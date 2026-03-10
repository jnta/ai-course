import json

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from qdrant_client import QdrantClient, models

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300

converter = DocumentConverter()
source = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(source)

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME), max_tokens=MAX_TOKENS
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True
)

chunks = list(chunker.chunk(result.document))

paper_title = "N/A"
paper_url = "N/A"

with open("./test_output/docling_paper_metadata.json", "r") as f:
    for line in f:
        doc = json.loads(line)
        for extraction in doc.get("extractions", []):
            extraction_class = extraction.get("extraction_class", "")
            extraction_text = extraction.get("extraction_text", "")
            
            if extraction_class == "title" and paper_title == "N/A":
                paper_title = extraction_text
            if extraction_class == "url" and paper_url == "N/A":
                paper_url = extraction_text
                
metadata_document_info = {
    "title": paper_title,
    "url": paper_url,
}

qdrant = QdrantClient(path="db/data")
qdrant.create_collection(
    collection_name="docling_paper",
    vector_config=models.VectorParams(size=qdrant.get_embedding_size(MODEL_NAME), distance=models.Distance.COSINE),
)

payload = []
embeddings = []
ids = []

for idx, chunk in enumerate(chunks):
    payload.append({
        "text": chunk.text,
        "metadata": metadata_document_info,
    })
    embeddings.append(models.Document(text=chunk.text).embed(model=MODEL_NAME))
    ids.append(idx)

qdrant.upload_collection(
    collection_name="docling_paper",
    vectors=embeddings,
    payload=payload,
    ids=ids
)

result = qdrant.query_points(
    collection_name="docling_paper",
    query=models.Document(
        text="What is the main contribution of the paper?",
        model=MODEL_NAME,
    )
).points

print("Top 3 relevant chunks:")
for point in result[:3]:    
    print(f"Score: {point.score:.4f}")
    print(f"Text: {point.payload['text'][:200]}...")
    print(f"Metadata: {point.payload['metadata']}")
    print("-" * 80)