"""
RAG Pipeline for Pharmaceutical Document Retrieval
===================================================
Projects 5–8: open-source LLM (Mistral) + page-level classifier
              + metadata-aware retrieval + Gradio frontend hooks.

Stack:
  * Loader      : PyMuPDF (page-level Documents) with Tesseract OCR fallback
  * Classifier  : Mistral 7B Instruct via LlamaIndex (deterministic JSON prompt)
  * Embeddings  : sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, local)
  * Indexing    : LlamaIndex VectorStoreIndex + SentenceSplitter
  * Retrieval   : Hybrid -> BM25 (keyword) + Vector (semantic), fused
  * Reranking   : SentenceTransformerRerank (cross-encoder/ms-marco-MiniLM-L-6-v2)
  * LLM         : Mistral 7B Instruct v0.2 (Q4_K_M GGUF, via llama-cpp-python)
                  — swappable with Gemini via LLM_BACKEND below.

Default backend is Mistral (open-source, no API key, runs locally on GPU).

Colab install (run once before importing this script):
    # base RAG stack
    !pip install -q llama-index pymupdf llama-index-embeddings-huggingface \
                    llama-index-retrievers-bm25 sentence-transformers gradio
    # OCR fallback for scanned PDFs
    !apt-get -qq install -y tesseract-ocr poppler-utils
    !pip install -q pytesseract pillow
    # Mistral via llama.cpp with CUDA
    !pip install --no-cache-dir llama-cpp-python==0.2.90 \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
    !pip install -q llama-index-llms-llama-cpp
    # download GGUF weights once (~4 GB)
    !wget -q https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O /content/mistral.gguf
    # optional, only if LLM_BACKEND="gemini"
    !pip install -q llama-index-llms-gemini
"""

import os
import re
import json
import math
import fitz  # PyMuPDF
from typing import List, Optional, Dict, Any

# -------------------------------------------------------------------
# 1. CONFIG
# -------------------------------------------------------------------
PDF_PATH       = "/home/titek/pharma-rag/documents/sample-sdf-document.pdf"
LLM_BACKEND    = "mistral"   # "mistral" | "gemini"

MISTRAL_GGUF_PATH   = "/home/titek/pharma-rag/models/mistral.gguf"
MISTRAL_CTX_WINDOW  = 4096
MISTRAL_MAX_TOKENS  = 512
MISTRAL_TEMPERATURE = 0.2
MISTRAL_GPU_LAYERS  = -1

GOOGLE_API_KEY = "AIzaSyBhQBfWBL9r-sl5tsif-gCq6ftsS1jXZoc"

CHUNK_SIZE         = 256
CHUNK_OVERLAP      = 40
TOP_K_RETRIEVE     = 6
TOP_K_RERANK       = 3
NUM_FUSION_QUERIES = 3

OCR_MIN_TEXT_CHARS = 30
ENABLE_CLASSIFIER  = True

DOC_TYPES = [
    "Cover Letter",
    "Certificate of Quality",
    "Packaging Specification",
    "BSE/TSE Declaration",
    "Material Description",
    "Supplier Qualification",
    "Chain of Custody",
    "Other",
]

INTERACTIVE = True
PROMPTS = [
    "What test methods were used for quality control?",
    "What are the storage conditions specified in the certificate?",
]

if LLM_BACKEND == "gemini":
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# -------------------------------------------------------------------
# 2. LLAMAINDEX IMPORTS
# -------------------------------------------------------------------
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

# Module-level flag — avoids touching Settings.llm (whose getter triggers
# LlamaIndex's lazy resolver and tries to instantiate OpenAI by default).
_MODELS_READY = False


# -------------------------------------------------------------------
# 3. LOAD PDF -> per-page Documents (with OCR fallback)
# -------------------------------------------------------------------
def _ocr_page(page) -> str:
    """OCR fallback for scanned pages. Returns '' if Tesseract isn't installed."""
    try:
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        return ""
    except Exception as e:
        print(f"[OCR] error on page: {e}")
        return ""


def load_pdf(pdf_path: str, progress_callback=None) -> List[Document]:
    """One LlamaIndex Document per page, with OCR fallback for scanned pages."""
    doc = fitz.open(pdf_path)
    documents: List[Document] = []
    n_ocr = 0
    for i, page in enumerate(doc):
        if progress_callback:
            progress_callback(i / max(len(doc), 1),
                              f"Loading page {i + 1}/{len(doc)}")
        text = page.get_text().strip()
        used_ocr = False
        if OCR_MIN_TEXT_CHARS and len(text) < OCR_MIN_TEXT_CHARS:
            ocr_text = _ocr_page(page)
            if ocr_text:
                text, used_ocr = ocr_text, True
                n_ocr += 1
        if not text:
            continue
        documents.append(Document(
            text=text,
            metadata={
                "file_name":   os.path.basename(pdf_path),
                "page_number": i + 1,
                "total_pages": len(doc),
                "ocr_used":    used_ocr,
            },
        ))
    doc.close()
    print(f"[Loader] Loaded {len(documents)} pages from {pdf_path}"
          + (f"  ({n_ocr} via OCR)" if n_ocr else ""))
    return documents


# -------------------------------------------------------------------
# 3b. PAGE CLASSIFIER (Project 7) — adds doc_type + doc_id metadata
# -------------------------------------------------------------------
_CLASSIFIER_PROMPT = """You are a pharmaceutical document classifier.
Classify the page below into EXACTLY ONE of these document types:
{types_list}

Definitions:
- Cover Letter: a formal letter (often "To Whom It May Concern") about a product, shipment, or storage condition.
- Certificate of Quality: contains lot numbers, manufacture/expiration dates, and test results (autoclave, gamma irradiation, flow rate, etc).
- Packaging Specification: describes packaging components, materials, part numbers, dimensions, configuration history.
- BSE/TSE Declaration: a declaration about animal-derived materials and transmissible spongiform encephalopathy compliance (USP <88>, EMA/410/01).
- Material Description: lists materials of construction, sterilization compatibility, physical properties.
- Supplier Qualification: documents a supplier's quality systems / certifications.
- Chain of Custody: tracks the handover or transfer of a sample / batch.
- Other: anything that does not clearly fit the above.

Reply with ONLY a JSON object in this exact format, no extra prose:
{{"doc_type": "<one of the labels above>"}}

If you are unsure, return {{"doc_type": "Other"}}.

PAGE CONTENT:
\"\"\"
{page_text}
\"\"\""""


def _parse_classifier_json(raw: str) -> Dict[str, str]:
    """Defensive JSON parse — Mistral wraps in markdown fences and adds prose."""
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        return {"doc_type": "Other"}
    try:
        obj = json.loads(m.group(0))
        dt = obj.get("doc_type", "Other")
        return {"doc_type": dt if dt in DOC_TYPES else "Other"}
    except json.JSONDecodeError:
        return {"doc_type": "Other"}


def classify_pages(documents: List[Document],
                   progress_callback=None) -> List[Document]:
    """Tag each page with doc_type (LLM); derive doc_id by grouping
    consecutive same-type pages. Mutates and returns documents."""
    if not ENABLE_CLASSIFIER or not documents:
        return documents

    types_list = "\n".join(f"  - {t}" for t in DOC_TYPES)
    prev_type = None
    doc_id = -1

    for i, d in enumerate(documents):
        if progress_callback:
            progress_callback(i / len(documents),
                              f"Classifying page {i + 1}/{len(documents)}")
        prompt = _CLASSIFIER_PROMPT.format(
            types_list=types_list,
            page_text=d.text[:1500],
        )
        try:
            raw = str(Settings.llm.complete(prompt))
        except Exception as e:
            print(f"[Classifier] LLM error on page {i + 1}: {e}")
            raw = '{"doc_type": "Other"}'
        result = _parse_classifier_json(raw)
        d.metadata["doc_type"] = result["doc_type"]

        if result["doc_type"] != prev_type:
            doc_id += 1
            prev_type = result["doc_type"]
        d.metadata["doc_id"] = doc_id
        d.metadata["is_doc_start"] = (
            i == 0 or documents[i - 1].metadata.get("doc_id") != doc_id
        )

    types_seen = sorted({d.metadata["doc_type"] for d in documents})
    print(f"[Classifier] Tagged {len(documents)} pages across "
          f"{doc_id + 1} logical document(s): {types_seen}")
    return documents


# -------------------------------------------------------------------
# 4. CONFIGURE MODELS (LLM + Embedding)
# -------------------------------------------------------------------
def _build_mistral_llm():
    if not os.path.exists(MISTRAL_GGUF_PATH):
        raise FileNotFoundError(
            f"Mistral GGUF not found at {MISTRAL_GGUF_PATH}. "
            "Run the wget command from the install header at the top of this file."
        )
    from llama_index.llms.llama_cpp import LlamaCPP
    return LlamaCPP(
        model_path=MISTRAL_GGUF_PATH,
        temperature=MISTRAL_TEMPERATURE,
        max_new_tokens=MISTRAL_MAX_TOKENS,
        context_window=MISTRAL_CTX_WINDOW,
        model_kwargs={"n_gpu_layers": MISTRAL_GPU_LAYERS, "verbose": False},
        verbose=False,
    )


def _build_gemini_llm():
    if GOOGLE_API_KEY.startswith("YOUR_"):
        raise SystemExit(
            "LLM_BACKEND='gemini' but GOOGLE_API_KEY is unset. "
            "Either set the key or switch LLM_BACKEND to 'mistral'."
        )
    from llama_index.llms.gemini import Gemini
    return Gemini(model="models/gemini-1.5-flash")


def configure_models():
    if LLM_BACKEND == "mistral":
        Settings.llm = _build_mistral_llm()
        llm_label = f"Mistral-7B-Instruct (GGUF Q4_K_M, ctx={MISTRAL_CTX_WINDOW})"
    elif LLM_BACKEND == "gemini":
        Settings.llm = _build_gemini_llm()
        llm_label = "Gemini-1.5-Flash"
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    global _MODELS_READY
    _MODELS_READY = True
    print(f"[Settings] {llm_label} + MiniLM + SentenceSplitter configured")


# -------------------------------------------------------------------
# 5. BUILD INDEX
# -------------------------------------------------------------------
def build_index(documents: List[Document]) -> VectorStoreIndex:
    index = VectorStoreIndex.from_documents(documents)
    n_nodes = len(list(index.docstore.docs.values()))
    print(f"[Index] Built VectorStoreIndex with {n_nodes} chunks")
    return index


# -------------------------------------------------------------------
# 6. HYBRID RETRIEVER (vector + BM25, query-fusion)
# -------------------------------------------------------------------
def build_hybrid_retriever(index: VectorStoreIndex):
    nodes = list(index.docstore.docs.values())
    vector_retriever = index.as_retriever(similarity_top_k=TOP_K_RETRIEVE)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=TOP_K_RETRIEVE
    )
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=TOP_K_RETRIEVE,
        num_queries=NUM_FUSION_QUERIES,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )


# -------------------------------------------------------------------
# 7. CROSS-ENCODER RERANKER
# -------------------------------------------------------------------
def build_reranker():
    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=TOP_K_RERANK,
    )


# -------------------------------------------------------------------
# 8. ASSEMBLE QUERY ENGINE  (citation-required prompt + optional
#    metadata filter for Project 7 routing)
# -------------------------------------------------------------------
_QA_PROMPT_TMPL = (
    "You are a precise pharmaceutical document assistant. Use ONLY the "
    "context below to answer the question. The context comes from real "
    "Certificates of Quality, BSE/TSE declarations, packaging "
    "specifications and similar regulated documents.\n\n"
    "Rules:\n"
    "  1. Quote facts exactly (lot numbers, dates, doses, temperatures).\n"
    "  2. After every fact, cite the source as (p. X) where X is the page number.\n"
    "  3. If the context does NOT contain the answer, reply exactly: "
    "\"I cannot find this in the provided documents.\"\n"
    "  4. Do NOT use outside knowledge. Do NOT speculate.\n\n"
    "----- CONTEXT -----\n{context_str}\n-------------------\n\n"
    "Question: {query_str}\n\nAnswer:"
)


def build_query_engine(index: VectorStoreIndex,
                       doc_type_filter: Optional[str] = None):
    """Hybrid retrieve -> rerank -> Mistral synth, with citation prompt.
    If doc_type_filter is given, both vector and BM25 are scoped to those pages."""
    from llama_index.core import PromptTemplate
    from llama_index.core.vector_stores import (
        MetadataFilters, MetadataFilter, FilterOperator,
    )

    if doc_type_filter and doc_type_filter not in (None, "", "All"):
        filters = MetadataFilters(filters=[
            MetadataFilter(key="doc_type", value=doc_type_filter,
                           operator=FilterOperator.EQ)
        ])
        all_nodes = list(index.docstore.docs.values())
        scoped_nodes = [n for n in all_nodes
                        if n.metadata.get("doc_type") == doc_type_filter]
        if not scoped_nodes:
            print(f"[QueryEngine] No chunks with doc_type='{doc_type_filter}'; "
                  "falling back to unfiltered retrieval.")
            fusion = build_hybrid_retriever(index)
        else:
            vector_retriever = index.as_retriever(
                similarity_top_k=TOP_K_RETRIEVE, filters=filters)
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=scoped_nodes, similarity_top_k=TOP_K_RETRIEVE)
            fusion = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=TOP_K_RETRIEVE,
                num_queries=NUM_FUSION_QUERIES,
                mode="reciprocal_rerank",
                use_async=False, verbose=False,
            )
    else:
        fusion = build_hybrid_retriever(index)

    qe = RetrieverQueryEngine.from_args(
        retriever=fusion,
        node_postprocessors=[build_reranker()],
    )
    qe.update_prompts({
        "response_synthesizer:text_qa_template": PromptTemplate(_QA_PROMPT_TMPL)
    })
    return qe


# -------------------------------------------------------------------
# 8b. CONFIDENCE + STRUCTURED prompt_func() FOR THE UI
# -------------------------------------------------------------------
def _score_to_prob(score: Optional[float]) -> float:
    """Cross-encoder logits -> 0..1 sigmoid probability."""
    if score is None:
        return 0.0
    try:
        return 1.0 / (1.0 + math.exp(-float(score)))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


def confidence_from_nodes(source_nodes) -> float:
    if not source_nodes:
        return 0.0
    probs = [_score_to_prob(getattr(n, "score", 0.0)) for n in source_nodes]
    return sum(probs) / len(probs)


def prompt_func(query_engine, query: str) -> Dict[str, Any]:
    """The single function the Gradio UI calls.
    Returns: { answer, sources[], confidence, chunks_used, fallback }."""
    if not query or not query.strip():
        return {"answer": "Please enter a question.",
                "sources": [], "confidence": 0.0,
                "chunks_used": 0, "fallback": True}

    response = query_engine.query(query)
    nodes = response.source_nodes

    if not nodes:
        return {"answer": "I cannot find this in the provided documents.",
                "sources": [], "confidence": 0.0,
                "chunks_used": 0, "fallback": True}

    sources = []
    for n in nodes:
        m = n.metadata or {}
        sources.append({
            "page":     m.get("page_number", "?"),
            "doc_type": m.get("doc_type", "Unknown"),
            "doc_id":   m.get("doc_id"),
            "score":    round(_score_to_prob(getattr(n, "score", 0.0)), 3),
            "preview":  (n.get_text() or "").replace("\n", " ")[:160],
        })

    return {
        "answer":      str(response).strip(),
        "sources":     sources,
        "confidence":  round(confidence_from_nodes(nodes), 3),
        "chunks_used": len(nodes),
        "fallback":    False,
    }


# -------------------------------------------------------------------
# 8c. PUBLIC SETUP — used by app.py (Gradio frontend)
# -------------------------------------------------------------------
def setup(pdf_path: str,
          doc_type_filter: Optional[str] = None,
          progress_callback=None):
    """One-shot index build for the UI. Returns (query_engine, documents)."""
    if not _MODELS_READY:
        configure_models()
    documents    = load_pdf(pdf_path, progress_callback=progress_callback)
    documents    = classify_pages(documents, progress_callback=progress_callback)
    index        = build_index(documents)
    query_engine = build_query_engine(index, doc_type_filter=doc_type_filter)
    return query_engine, documents


# -------------------------------------------------------------------
# 9. ANSWER A SINGLE QUERY (CLI helper)
# -------------------------------------------------------------------
def answer_query(query_engine, prompt: str) -> None:
    print("\n" + "=" * 72)
    print(f"PROMPT: {prompt}")
    print("=" * 72)
    result = prompt_func(query_engine, prompt)
    print(result["answer"])
    print(f"\n[confidence: {result['confidence']:.2f}  "
          f"chunks: {result['chunks_used']}]")
    if result["sources"]:
        print("--- Source chunks used ---")
        for j, s in enumerate(result["sources"], 1):
            print(f"  [{j}] p.{s['page']}  ({s['doc_type']})  "
                  f"score={s['score']:.3f}  {s['preview']}...")


# -------------------------------------------------------------------
# 10. MAIN ENTRY (CLI) — interactive prompt loop
# -------------------------------------------------------------------
def run_pipeline():
    """Interactive CLI: build the index once, then chat in a loop."""
    if LLM_BACKEND == "mistral" and not os.path.exists(MISTRAL_GGUF_PATH):
        raise SystemExit(
            f"LLM_BACKEND='mistral' but no GGUF file at {MISTRAL_GGUF_PATH}.\n"
            "Run the wget command from the install header at the top of this file."
        )
    if LLM_BACKEND == "gemini" and GOOGLE_API_KEY.startswith("YOUR_"):
        raise SystemExit(
            "LLM_BACKEND='gemini' but GOOGLE_API_KEY is unset. "
            "Either set the key or switch LLM_BACKEND to 'mistral'."
        )

    query_engine, _docs = setup(PDF_PATH)

    if not INTERACTIVE:
        for prompt in PROMPTS:
            answer_query(query_engine, prompt)
        return

    print("\n" + "=" * 72)
    print(" Pharma RAG chatbot ready.")
    print(" Ask anything about the document. Type 'exit' (or empty line) to quit.")
    print("=" * 72)

    while True:
        try:
            prompt = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt or prompt.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        try:
            answer_query(query_engine, prompt)
        except Exception as e:
            print(f"[Error answering query] {e}")


if __name__ == "__main__":
    run_pipeline()
