import nltk

nltk.download('punkt_tab')

text = "Hello, world! This is a test."

word_tokens = nltk.word_tokenize(text, language='english')
print(word_tokens)

sentece_tokens = nltk.sent_tokenize(text, language='english')
print(sentece_tokens)