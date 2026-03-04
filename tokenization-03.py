import os
import shutil

import nltk
from whoosh.fields import Schema, ID, TEXT
from whoosh.index import create_in
from whoosh.qparser import QueryParser

nltk.download('stopwords')

documents = [
    "Machine learning is a field of artificial intelligence.",
    "Artificial intelligence is a branch of computer science.",
    "Natural language processing is a subfield of artificial intelligence.",
    "This document is unrelated to the others.",
    "Another unrelated document.",
]

query = "machine not learning"

def preprocess(text):
    word_tokens = nltk.word_tokenize(text.lower(), language='english')
    stopwords = set(nltk.corpus.stopwords.words('english')) - {'not', 'or', 'and'}
    return [word for word in word_tokens if word.isalnum() and word not in stopwords]

if os.path.exists('index_dir'):
    shutil.rmtree('index_dir')
os.mkdir('index_dir')

schema = Schema(title=ID(stored=True, unique=True), content=TEXT(stored=True))

index = create_in('index_dir', schema)
writer = index.writer()
for i, doc in enumerate(documents):
    writer.add_document(title=str(i), content=doc)
writer.commit()

query = "artificial AND intelligence"
parser = QueryParser("content", index.schema)
parsed_query = parser.parse(query)
with index.searcher() as searcher:
    results = searcher.search(parsed_query, limit=None)
    print(f"Top {len(results)} most similar documents to the query '{query}':")
    for result in results:
        print(f"Document {result['title']}: {result['content']}")