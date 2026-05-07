# Pharma Document Intelligence

**Grounded answers for regulated documents.**

A production-grade Retrieval-Augmented Generation (RAG) pipeline designed for pharmaceutical document processing. Extract, classify, and query technical information from pharma documents with an intuitive web interface.

## Features

- **🔍 Intelligent Document Retrieval** - Hybrid search combining BM25 keyword matching with semantic vector search
- **🧠 Local LLM Processing** - Mistral 7B Instruct running locally (no API keys required)
- **📄 Multi-Format Support** - Handles both digital PDFs and scanned documents with OCR fallback
- **🏷️ Automatic Classification** - Deterministic page-level document classification
- **⚡ Smart Reranking** - Cross-encoder reranking for improved result relevance
- **🎨 Web Interface** - Clean Gradio-based UI for easy document querying

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Document Processing** | PyMuPDF (fitz) with Tesseract OCR fallback |
| **Classification** | Mistral 7B Instruct via LlamaIndex |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Indexing** | LlamaIndex VectorStoreIndex |
| **Retrieval** | Hybrid BM25 + Vector Search |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **LLM** | Mistral 7B Instruct v0.2 (Q4_K_M GGUF) |
| **Frontend** | Gradio |

## Installation

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA for llama-cpp-python acceleration)
- ~4 GB disk space for model weights

### Setup

1. Clone the repository:
```bash
cd pharma-rag
```

2. Install dependencies:
```bash
pip install -q llama-index pymupdf llama-index-embeddings-huggingface \
                llama-index-retrievers-bm25 sentence-transformers gradio
```

3. Install OCR support (optional, for scanned PDFs):
```bash
apt-get install -y tesseract-ocr poppler-utils
pip install -q pytesseract pillow
```

4. Install Mistral LLM with CUDA support:
```bash
pip install --no-cache-dir llama-cpp-python==0.2.90 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
pip install -q llama-index-llms-llama-cpp
```

5. Download model weights (~4 GB):
```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    -O models/mistral.gguf
```

## Usage

### Start the Web Interface

```bash
python app.py
```

The Gradio interface will launch (typically at `http://localhost:7860`).

### Upload & Query

1. Upload a pharmaceutical document (PDF)
2. Select document type if needed
3. Ask questions about the document
4. Get grounded, sourced answers

### Example Queries

```
- "What test methods were used for quality control?"
- "What are the storage conditions specified in the certificate?"
- "Was this product gamma-irradiated, and at what dose?"
- "What is the lot number on the certificate?"
- "Does this product comply with USP <88> Class VI?"
```

## Project Structure

```
pharma-rag/
├── app.py                           # Gradio frontend
├── rag_pipeline.py                  # RAG pipeline engine
├── documents/
│   └── sample-sdf-document.pdf     # Example pharmaceutical document
├── models/
│   └── mistral.gguf                # Mistral LLM weights
└── README.md                        # This file
```

## Configuration

Edit `rag_pipeline.py` to customize:

```python
PDF_PATH            = "/path/to/document.pdf"      # Document path
LLM_BACKEND         = "mistral"     # "mistral" or "gemini"
MISTRAL_GGUF_PATH   = "/path/to/mistral.gguf"     # Model weights
MISTRAL_CTX_WINDOW  = 4096          # Context window size
```

### Alternative Backend: Google Gemini

To use Gemini instead of Mistral:

```bash
pip install -q llama-index-llms-gemini
```

Set in `rag_pipeline.py`:
```python
LLM_BACKEND = "gemini"
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

## Performance Tuning

- **GPU Acceleration**: Install CUDA-enabled llama-cpp-python for 10x+ speedup
- **Chunk Size**: Adjust `SentenceSplitter` parameters in `rag_pipeline.py`
- **Retrieval Top-K**: Increase retrieved documents for higher recall
- **Reranking**: Trade-off between speed and relevance

## Troubleshooting

### OCR Not Working
```bash
# Ensure tesseract is installed
apt-get install tesseract-ocr
# Verify Tesseract path in rag_pipeline.py
```

### Out of Memory
- Reduce `chunk_size` in SentenceSplitter
- Use smaller model (e.g., 3B instead of 7B)
- Enable page-level processing for large documents

### Slow Inference
- Enable GPU acceleration (CUDA)
- Reduce context window
- Use quantized model (Q4_K_M is already quantized)

## License

MIT 

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- All dependencies are documented
- Tests pass for document processing

## Support

For issues or questions, please open an issue on GitHub.
