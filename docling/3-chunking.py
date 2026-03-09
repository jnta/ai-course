from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

converter = DocumentConverter()
source = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(source)
chunker = HierarchicalChunker()
chunks = list(chunker.chunk(result.document))
print(chunks[0].text)
