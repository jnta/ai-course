import nltk;
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

preprocessed_documents = [" ".join(preprocess(doc)) for doc in documents]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

query = "Machine learning and artificial intelligence"

def search_tfidf(query, vectorizer, tfidf_matrix):
    preprocessed_query = " ".join(preprocess(query))
    query_vector = vectorizer.transform([preprocessed_query])
    similarity_scores = cosine_similarity(tfidf_matrix, query_vector).flatten()
    return sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)

similarity_scores = search_tfidf(query, vectorizer, tfidf_matrix)

print(f"Top 3 most similar documents to the query '{query}':")
for doc_index, score in similarity_scores[:3]:
    print(f"Document {doc_index}: {documents[doc_index]}")