import sys
print("Python executable:", sys.executable)
try:
    import dotenv
    print("dotenv avail")
except ImportError:
    print("dotenv missing")

try:
    import docling.chunking
    print("docling avail")
except ImportError:
    print("docling missing")

try:
    import langchain_milvus
    print("milvus avail")
except ImportError:
    print("milvus missing")
