import nltk
import numpy as np
from rank_bm25 import BM25Okapi

documents = [
    "Machine learning is a field of artificial intelligence.",
    "Artificial intelligence is a branch of computer science.",
    "Natural language processing is a subfield of artificial intelligence.",
    "This document is unrelated to the others.",
    "Another unrelated document.",
]

def preprocess(text):
    word_tokens = nltk.word_tokenize(text.lower(), language='english')
    return [word for word in word_tokens if word.isalnum()]

tokenized_documents = [preprocess(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_documents)
query = "machine learning"

def search_bm25(query, bm25):
    tokenized_query = preprocess(query)
    return bm25.get_scores(tokenized_query)

similarity_scores = search_bm25(query, bm25)
print(f"Top 3 most similar documents to the query '{query}':")
top_indices = np.argsort(similarity_scores)[::-1][:3]
for index in top_indices:
    print(f"Document {index}: {documents[index]}")