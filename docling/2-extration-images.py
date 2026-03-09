from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem
import os

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0
pipeline_options.generate_picture_images = True

converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
result = converter.convert("https://arxiv.org/pdf/2408.09869")

os.makedirs("output_images", exist_ok=True)

picture_counter = 0
for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        picture_counter += 1
        with open(f"output_images/picture_{picture_counter}.bin", "wb") as f:
            element.get_image(result.document).save(f, format="PNG")