from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(":memory:")
model = SentenceTransformer('all-MiniLM-L6-v2')
client_groq = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

documents = [
    "Machine learning is a field of artificial intelligence.",
    "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that allow computers to learn from data without being explicitly programmed.",
    "Artificial intelligence is a branch of computer science.",
    "Natural language processing is a subfield of artificial intelligence.",
    "This document is unrelated to the others.",
    "Another unrelated document.",
]

vector_size = model.get_sentence_embedding_dimension()

client.recreate_collection(
    collection_name="ml_docs",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

points = []
for i, doc in enumerate(documents):
    points.append(
        PointStruct(
            id=i,
            vector=model.encode(doc).tolist(),
            payload={"text": doc}
        )
    )

client.upsert(collection_name="ml_docs", points=points)

def retrieve_documents(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    search_result = client.query_points(
        collection_name="ml_docs",
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )
    return [(hit.payload["text"], hit.score) for hit in search_result.points]

def generate_response(query, retrieved_docs):
    context = "\n".join([doc for doc, _ in retrieved_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return response.choices[0].message.content

def rag(query):
    retrieved_docs = retrieve_documents(query)
    answer = generate_response(query, retrieved_docs)
    return answer, retrieved_docs

answer, retrieved_docs = rag("What is machine learning?")
print(answer)
for doc, similarity in retrieved_docs:
    print(f"Document: {doc}")
    print(f"Similarity: {similarity}")