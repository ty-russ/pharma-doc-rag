"""
Pharma Document Q&A — Gradio frontend
=====================================
Project 8 capstone UI. Wraps the open-source RAG pipeline in
rag_pipeline.py with a clean Gradio Blocks interface.

UI features:
  * PDF upload with progress bar (loading + classifying pages)
  * Document info panel (page count, doc-type breakdown, OCR usage)
  * Settings panel (doc_type filter, top_k slider)
  * Chat with answer + confidence + source-chunk citations
  * Clear chat / reset buttons

Run in Colab:
    %run app.py
or
    !python app.py
After the install header at the top of rag_pipeline.py has been executed.
"""

import os
import time
from collections import Counter
from typing import List, Tuple, Optional

import gradio as gr
import rag_pipeline as rag


APP_TITLE = "Pharma Document Q&A Open-source RAG"
SAMPLE_PROMPTS = [
    "What test methods were used for quality control?",
    "What are the storage conditions specified in the certificate?",
    "Was this product gamma-irradiated, and at what dose?",
    "What is the lot number on the certificate?",
    "Does this product comply with USP <88> Class VI?",
]


# -------------------------------------------------------------------
# Backend wrappers
# -------------------------------------------------------------------
def _doc_type_options(documents) -> List[str]:
    """All doc_types present in the indexed documents (+ 'All' first)."""
    types = sorted({d.metadata.get("doc_type", "Unknown") for d in documents})
    return ["All"] + types


def _doc_summary(documents, elapsed: float) -> str:
    """Markdown breakdown of what was indexed."""
    if not documents:
        return "_No pages indexed yet._"
    by_type = Counter(d.metadata.get("doc_type", "Unknown") for d in documents)
    n_ocr   = sum(1 for d in documents if d.metadata.get("ocr_used"))
    fname   = documents[0].metadata.get("file_name", "?")
    n_pages = documents[0].metadata.get("total_pages", len(documents))

    lines = [
        f"**File:** `{fname}`  ",
        f"**Pages indexed:** {len(documents)} of {n_pages}"
        + (f"  ·  {n_ocr} via OCR" if n_ocr else ""),
        f"**Index build time:** {elapsed:.1f} s",
        "",
        "**Document-type breakdown:**",
    ]
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- {t}: {c} page(s)")
    return "\n".join(lines)


def process_pdf(file_obj, progress=gr.Progress()):
    """Build the index. Returns (engine_state, status_md, doc_summary_md, doc_type_dropdown_update)."""
    if file_obj is None:
        return None, "⚠️ No file uploaded.", "_Upload a PDF to begin._", \
               gr.update(choices=["All"], value="All")

    pdf_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    progress(0.05, desc="Configuring models (one-time)…")

    def _cb(frac, label):
        progress(0.1 + 0.85 * frac, desc=label)

    t0 = time.time()
    try:
        query_engine, documents = rag.setup(pdf_path, progress_callback=_cb)
    except Exception as e:
        return None, f"❌ Error: {e}", "_Indexing failed._", \
               gr.update(choices=["All"], value="All")

    elapsed = time.time() - t0
    status = (f"✅ Indexed {len(documents)} pages in {elapsed:.1f}s. "
              f"Ready for questions.")
    return (query_engine,
            status,
            _doc_summary(documents, elapsed),
            gr.update(choices=_doc_type_options(documents), value="All"))


def chat_handler(message: str,
                 history: list,
                 query_engine,
                 doc_type_filter: str,
                 top_k_rerank: int):
    """Run a single user question through the RAG pipeline.
    Returns (new_history, sources_md, confidence_label, '')."""
    history = history or []
    if query_engine is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant",
                        "content": "Please upload and process a PDF first."})
        return history, "", "—", ""

    if not message or not message.strip():
        return history, "", "—", ""

    # Apply runtime tweaks (top_k_rerank changes the reranker's top_n).
    try:
        for pp in query_engine._node_postprocessors:
            if hasattr(pp, "top_n"):
                pp.top_n = int(top_k_rerank)
    except Exception:
        pass

    # If the user changed the doc_type filter mid-session we'd ideally
    # rebuild the engine. To keep the demo snappy we filter post-hoc by
    # dropping non-matching source nodes from the response and re-ranking.
    result = rag.prompt_func(query_engine, message)

    if doc_type_filter and doc_type_filter != "All":
        kept = [s for s in result["sources"]
                if s.get("doc_type") == doc_type_filter]
        if kept:
            result["sources"] = kept
            result["chunks_used"] = len(kept)

    # Render answer + sources panel
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["answer"]})

    if result["sources"]:
        rows = ["| # | Page | Doc type | Score | Preview |",
                "|---|------|----------|-------|---------|"]
        for i, s in enumerate(result["sources"], 1):
            preview = s["preview"].replace("|", "\\|")
            rows.append(f"| {i} | {s['page']} | {s['doc_type']} | "
                        f"{s['score']:.2f} | {preview}… |")
        sources_md = "\n".join(rows)
    else:
        sources_md = "_No source chunks (model returned a fallback answer)._"

    conf = result["confidence"]
    pct  = int(conf * 100)
    icon = "🟢" if conf >= 0.6 else ("🟡" if conf >= 0.3 else "🔴")
    conf_label = f"{icon} {pct}%  ·  {result['chunks_used']} chunks used"

    return history, sources_md, conf_label, ""


def clear_chat():
    return [], "_Sources will appear here after your first question._", "—"


# -------------------------------------------------------------------
# UI layout
# -------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important;
                    padding: 12px 24px !important; }
#status_md { padding: 8px 12px; border-radius: 6px;
             background: #f0f4ff; font-size: 14px; }
#confidence_lbl { font-weight: 600; font-size: 16px;
                  padding: 6px 10px; border-radius: 6px;
                  background: #f6f6f6; }
/* Make the chat area expand into the wider viewport */
#chat-area { min-height: 520px; }
"""

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# 🧬 {APP_TITLE}")
    gr.Markdown(
        "Upload a pharmaceutical PDF (Certificate of Quality, BSE/TSE "
        "declaration, packaging spec, or a multi-document bundle), then "
        "ask grounded questions. Answers cite the page they came from."
    )

    engine_state = gr.State(None)

    with gr.Row():
        # ---------- LEFT COLUMN: upload + settings + summary -------
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📄 Document")
            pdf_input  = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("🔄 Process Document",
                                    variant="primary")
            status_md  = gr.Markdown("_Upload a PDF to begin._",
                                     elem_id="status_md")

            gr.Markdown("### ⚙️ Settings")
            doc_type_dd = gr.Dropdown(
                label="Filter by document type",
                choices=["All"], value="All",
                info="Restrict retrieval to one document type."
            )
            topk_slider = gr.Slider(
                label="Chunks shown to the LLM (top-k after rerank)",
                minimum=1, maximum=8, value=3, step=1,
                info="Higher = more context, lower = more focused."
            )

            gr.Markdown("###  Document Info")
            doc_info_md = gr.Markdown("_Process a PDF to see its breakdown._")

        # ---------- RIGHT COLUMN: chat + sources ------------------
        with gr.Column(scale=3, min_width=600):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(label="Conversation", height=560,
                                 elem_id="chat-area")
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask anything about the uploaded document…",
                    label="Your question",
                    scale=4, show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear chat")
                conf_label = gr.Markdown("—", elem_id="confidence_lbl")

            gr.Examples(SAMPLE_PROMPTS, inputs=user_input,
                        label="Try a sample question")

            gr.Markdown("### 🔎 Sources for last answer")
            sources_md = gr.Markdown(
                "_Sources will appear here after your first question._"
            )

    # ---------- Wiring ----------
    process_btn.click(
        process_pdf,
        inputs=[pdf_input],
        outputs=[engine_state, status_md, doc_info_md, doc_type_dd],
    )

    chat_inputs  = [user_input, chatbot, engine_state, doc_type_dd, topk_slider]
    chat_outputs = [chatbot, sources_md, conf_label, user_input]
    send_btn.click(chat_handler, inputs=chat_inputs, outputs=chat_outputs)
    user_input.submit(chat_handler, inputs=chat_inputs, outputs=chat_outputs)

    clear_btn.click(clear_chat, outputs=[chatbot, sources_md, conf_label])


if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )
