# LlamaIndex ZeusDB Integration

ZeusDB vector database integration for LlamaIndex. Connect LlamaIndex's RAG framework with high-performance, enterprise-grade vector database.

## Features

- **Production Ready**: Built for enterprise-scale RAG applications
- **Advanced Filtering**: Comprehensive metadata filtering with complex operators  
- **MMR Support**: Maximal Marginal Relevance for diverse result sets
- **Async Helpers**: Async wrappers for add, query, delete, and clear

## Installation

```bash
pip install llama-index-vector-stores-zeusdb
```

## Quick Start

```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Set up embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Create ZeusDB vector store
vector_store = ZeusDBVectorStore(
    dim=1536,  # OpenAI embedding dimension
    distance="cosine",
    index_type="hnsw"
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create documents
documents = [
    Document(text="ZeusDB is a high-performance vector database."),
    Document(text="LlamaIndex provides RAG capabilities."),
    Document(text="Vector search enables semantic similarity.")
]

# Create index and store documents
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is ZeusDB?")
print(response)
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dim` | Vector dimension | Required |
| `distance` | Distance metric (`cosine`, `l2`, `l1`) | `cosine` |
| `index_type` | Index type (`hnsw`) | `hnsw` |
| `m` | HNSW connectivity parameter | 16 |
| `ef_construction` | HNSW build-time search depth | 200 |

## Documentation

For comprehensive guides, advanced examples, and configuration options, visit:

**[ZeusDB LlamaIndex Documentation](https://docs.zeusdb.com/integrations/llamaindex)**

## Requirements

- Python 3.10+
- llama-index-core >= 0.13.6
- zeusdb >= 0.0.8

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](https://github.com/ZeusDB/.github/CONTRIBUTING.md) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/ZeusDB/llama-index-vector-stores-zeusdb/issues)
- **Documentation**: [docs.zeusdb.com](https://docs.zeusdb.com)
- **Email**: [contact@zeusdb.com](mailto:contact@zeusdb.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
