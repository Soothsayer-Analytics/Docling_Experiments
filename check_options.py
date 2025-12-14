
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

print("PdfPipelineOptions attributes:")
print([a for a in dir(PdfPipelineOptions()) if not a.startswith('_')])

print("\nTableStructureOptions attributes:")
print([a for a in dir(TableStructureOptions()) if not a.startswith('_')])
