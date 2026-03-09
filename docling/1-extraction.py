from docling.document_converter import DocumentConverter

converter = DocumentConverter()
source = "https://arxiv.org/pdf/2408.09869"
result = converter.convert(source)
print(result.document.export_to_markdown())