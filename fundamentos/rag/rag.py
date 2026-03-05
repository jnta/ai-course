from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(
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

doc_embeddings = model.encode(documents)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve_documents(query, k=3):
    query_embedding = model.encode([query])[0]
    similarity_scores = []
    for i, doc_embedding in enumerate(doc_embeddings):
        similarity_scores.append((i, cosine_similarity(query_embedding, doc_embedding)))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return [(documents[index], score) for index, score in similarity_scores[:k]]


def generate_response(query, retrieved_docs):
    context = "\n".join([doc for doc,_ in retrieved_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(
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
    