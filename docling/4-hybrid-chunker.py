from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300

converter = DocumentConverter()
source = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(source)

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_NAME), max_tokens=MAX_TOKENS
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True
)

chunks = list(chunker.chunk(result.document))
for chunk in chunks[:5]:
    print(f"Chunk with {tokenizer.count_tokens(chunk.text)} tokens: {chunk.text[:200]}...")