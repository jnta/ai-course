import nltk

nltk.download('punkt_tab')

text = "Hello, world! This is a test."

word_tokens = nltk.word_tokenize(text, language='english')
print(word_tokens)

sentece_tokens = nltk.sent_tokenize(text, language='english')
print(sentece_tokens)

def preprocess(text):
    word_tokens = nltk.word_tokenize(text.lower(), language='english')
    return [word for word in word_tokens if word.isalnum()]

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]

preprocessed_documents = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed_documents)