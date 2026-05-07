"""
Pharma Document Intelligence — Gradio frontend
==============================================
Production-styled UI wrapping the open-source RAG pipeline.

Run: python app.py     (after rag_pipeline.py is in the same folder)
"""

import time
from collections import Counter
from typing import List

import gradio as gr
import rag_pipeline as rag

PRODUCT_NAME    = "Pharma Document Intelligence"
PRODUCT_TAGLINE = "Grounded answers for regulated documents."
SAMPLE_PROMPTS = [
    "What test methods were used for quality control?",
    "What are the storage conditions specified in the certificate?",
    "Was this product gamma-irradiated, and at what dose?",
    "What is the lot number on the certificate?",
    "Does this product comply with USP <88> Class VI?",
]


# ------------------------------------------------------------------
# Backend wrappers
# ------------------------------------------------------------------
def _doc_type_options(documents) -> List[str]:
    types = sorted({d.metadata.get("doc_type", "Unknown") for d in documents})
    return ["All"] + types


def _doc_summary_html(documents, elapsed: float) -> str:
    if not documents:
        return ""
    by_type = Counter(d.metadata.get("doc_type", "Unknown") for d in documents)
    n_ocr   = sum(1 for d in documents if d.metadata.get("ocr_used"))
    fname   = documents[0].metadata.get("file_name", "—")
    n_pages = documents[0].metadata.get("total_pages", len(documents))

    rows = "".join(
        f'<div class="kv-row"><span>{t}</span><span class="kv-num">{c}</span></div>'
        for t, c in sorted(by_type.items(), key=lambda x: -x[1])
    )
    ocr_part = f" · {n_ocr} OCR" if n_ocr else ""
    return f"""
    <div class="info-card">
      <div class="info-meta">
        <div><span class="lbl">File</span><code>{fname}</code></div>
        <div><span class="lbl">Pages</span>{len(documents)} of {n_pages}{ocr_part}</div>
        <div><span class="lbl">Index time</span>{elapsed:.1f}s</div>
      </div>
      <div class="kv-divider"></div>
      <div class="kv-title">Document types</div>
      {rows}
    </div>
    """


def process_pdf(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return None, '<div class="status status-warn">No file uploaded.</div>', "", \
               gr.update(choices=["All"], value="All")

    pdf_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    progress(0.05, desc="Configuring models")
    def _cb(frac, label):
        progress(0.1 + 0.85 * frac, desc=label)

    t0 = time.time()
    try:
        query_engine, documents = rag.setup(pdf_path, progress_callback=_cb)
    except Exception as e:
        return None, f'<div class="status status-err">Indexing failed: {e}</div>', "", \
               gr.update(choices=["All"], value="All")

    elapsed = time.time() - t0
    status = (f'<div class="status status-ok">Indexed {len(documents)} pages '
              f'in {elapsed:.1f}s. Ready.</div>')
    return (query_engine, status,
            _doc_summary_html(documents, elapsed),
            gr.update(choices=_doc_type_options(documents), value="All"))


def chat_handler(message, history, query_engine, doc_type_filter, top_k_rerank):
    history = history or []
    if query_engine is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant",
                        "content": "Please upload and index a PDF first."})
        return history, "", '<div class="conf-empty">—</div>', ""

    if not message or not message.strip():
        return history, "", '<div class="conf-empty">—</div>', ""

    try:
        for pp in query_engine._node_postprocessors:
            if hasattr(pp, "top_n"):
                pp.top_n = int(top_k_rerank)
    except Exception:
        pass

    result = rag.prompt_func(query_engine, message)

    if doc_type_filter and doc_type_filter != "All":
        kept = [s for s in result["sources"]
                if s.get("doc_type") == doc_type_filter]
        if kept:
            result["sources"] = kept
            result["chunks_used"] = len(kept)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["answer"]})

    if result["sources"]:
        rows = "".join(
            f"""<tr>
              <td class="src-rank">{i}</td>
              <td class="src-page">p.{s['page']}</td>
              <td class="src-type">{s['doc_type']}</td>
              <td class="src-score">{s['score']:.2f}</td>
              <td class="src-prev">{s['preview'].replace('<','&lt;').replace('>','&gt;')}…</td>
            </tr>"""
            for i, s in enumerate(result["sources"], 1)
        )
        sources_html = (
            '<table class="src-tbl"><thead>'
            '<tr><th>#</th><th>Page</th><th>Type</th><th>Score</th><th>Preview</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )
    else:
        sources_html = '<div class="src-empty">No source chunks for this answer.</div>'

    conf = result["confidence"]
    pct  = int(conf * 100)
    cls  = "conf-high" if conf >= 0.6 else ("conf-mid" if conf >= 0.3 else "conf-low")
    conf_html = (f'<div class="conf {cls}"><span class="conf-dot"></span>'
                 f'<span class="conf-pct">{pct}%</span>'
                 f'<span class="conf-meta">{result["chunks_used"]} chunks</span></div>')

    return history, sources_html, conf_html, ""


def clear_chat():
    return [], "", '<div class="conf-empty">—</div>'


# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F7F8FA",
    body_text_color="#0F172A",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F7F8FA",
    border_color_primary="#E5EAF0",
    block_border_width="1px",
    block_radius="8px",
    block_label_text_size="13px",
    block_label_text_weight="500",
    block_title_text_weight="600",
    button_primary_background_fill="#1F3864",
    button_primary_background_fill_hover="#162A4D",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#1F3864",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F1F4F9",
    button_secondary_text_color="#1F3864",
    button_secondary_border_color="#D9DEE7",
    input_background_fill="#FFFFFF",
    input_border_color="#D9DEE7",
    input_border_color_focus="#1F3864",
)


CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important;
                    padding: 0 !important; margin: 0 !important; }
.gradio-container > .main { padding: 0 !important; }

/* Top bar */
#topbar { background:#0F172A; color:#fff; padding:14px 28px;
          display:flex; align-items:center; justify-content:space-between;
          border-bottom:1px solid #1F2A40; }
#topbar, #topbar * { color:#FFFFFF; }
#topbar .brand { font-size:16px; font-weight:600; letter-spacing:-0.01em;
                 color:#FFFFFF !important; margin-bottom:2px; }
#topbar .tag   { color:#9DA9BD !important; font-size:13px; }
#topbar .badge { font-size:11px; color:#9DA9BD; border:1px solid #324A6D;
                 padding:2px 8px; border-radius:99px; letter-spacing:0.04em;
                 text-transform:uppercase; }

/* Body wrapper */
#body { padding:20px 28px 32px; }

/* Section labels */
.sec-label { font-size:11px; font-weight:600; letter-spacing:0.08em;
             text-transform:uppercase; color:#5A6B85; margin:14px 0 8px; }

/* Status pill under upload */
.status { font-size:13px; padding:8px 12px; border-radius:6px;
          border:1px solid transparent; line-height:1.4; }
.status-ok   { background:#EAF7F1; color:#1A6E4B; border-color:#C9E8D8; }
.status-warn { background:#FFF4E0; color:#7A5410; border-color:#F2DBA8; }
.status-err  { background:#FDECEC; color:#902323; border-color:#F0BCBC; }

/* Document-info card */
.info-card { font-size:13px; }
.info-card .info-meta { display:grid; gap:6px; }
.info-card .info-meta .lbl { display:inline-block; min-width:80px;
                              color:#6E7B92; font-size:11px;
                              text-transform:uppercase; letter-spacing:0.06em; }
.info-card code { background:#F1F4F9; padding:1px 6px; border-radius:4px;
                  font-size:12px; }
.kv-divider { height:1px; background:#E5EAF0; margin:12px 0; }
.kv-title { font-size:11px; font-weight:600; letter-spacing:0.06em;
            text-transform:uppercase; color:#6E7B92; margin-bottom:6px; }
.kv-row { display:flex; justify-content:space-between;
          padding:4px 0; font-size:13px; border-bottom:1px dashed #EEF1F6; }
.kv-row:last-child { border-bottom:none; }
.kv-row .kv-num { color:#5A6B85; font-variant-numeric:tabular-nums; }

/* Confidence pill */
.conf { display:inline-flex; align-items:center; gap:8px;
        padding:6px 12px; border-radius:99px; font-size:13px;
        background:#F1F4F9; border:1px solid #E5EAF0; }
.conf-dot { width:8px; height:8px; border-radius:50%; background:#9DA9BD; }
.conf-pct { font-weight:600; font-variant-numeric:tabular-nums; }
.conf-meta { color:#6E7B92; font-size:12px; }
.conf-high .conf-dot { background:#2BA48F; }
.conf-high { background:#EAF7F1; border-color:#C9E8D8; color:#1A6E4B; }
.conf-mid  .conf-dot { background:#D69A2E; }
.conf-mid  { background:#FFF4E0; border-color:#F2DBA8; color:#7A5410; }
.conf-low  .conf-dot { background:#C44545; }
.conf-low  { background:#FDECEC; border-color:#F0BCBC; color:#902323; }
.conf-empty { color:#9DA9BD; font-size:13px; padding:6px 12px; }

/* Sources table */
.src-tbl { width:100%; border-collapse:collapse; font-size:13px;
           font-variant-numeric:tabular-nums; }
.src-tbl th { text-align:left; padding:8px 10px; font-weight:600;
              font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
              color:#6E7B92; border-bottom:1px solid #E5EAF0; }
.src-tbl td { padding:10px; border-bottom:1px solid #EEF1F6;
              vertical-align:top; }
.src-tbl tr:last-child td { border-bottom:none; }
.src-rank { color:#9DA9BD; width:24px; }
.src-page { font-weight:600; color:#1F3864; width:50px; }
.src-type { color:#5A6B85; width:170px; }
.src-score{ color:#5A6B85; width:60px; }
.src-prev { color:#0F172A; }
.src-empty { font-size:13px; color:#9DA9BD; padding:14px 0; }

/* Chat area */
#chat-area { border:1px solid #E5EAF0 !important; border-radius:10px !important;
             background:#fff !important; }

/* Suggested-query chips */
.examples-holder button {
  background:#fff !important; border:1px solid #E1E6EE !important;
  color:#1F3864 !important; font-weight:500 !important;
  border-radius:99px !important; padding:6px 14px !important;
  font-size:13px !important; box-shadow:none !important;
}
.examples-holder button:hover { background:#F1F4F9 !important; }

/* Tighten markdown headings */
h1, h2, h3 { letter-spacing:-0.01em; }

/* Hide gradio default footer */
footer { display:none !important; }
"""


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
TOPBAR_HTML = f"""
<div id="topbar">
  <div>
    <div class="brand">{PRODUCT_NAME}</div>
    <div class="tag">{PRODUCT_TAGLINE}</div>
  </div>
  <div class="badge">v1.0 · Mistral 7B · Hybrid RAG</div>
</div>
"""

with gr.Blocks(title=PRODUCT_NAME, theme=THEME, css=CSS) as demo:
    gr.HTML(TOPBAR_HTML)
    engine_state = gr.State(None)

    with gr.Row(elem_id="body"):
        # ---- Left rail -----------------------------------------------
        with gr.Column(scale=1, min_width=320):
            gr.HTML('<div class="sec-label">Document</div>')
            pdf_input  = gr.File(label="Upload PDF", show_label=False,
                                 file_types=[".pdf"])
            process_btn = gr.Button("Index document", variant="primary")
            status_md  = gr.HTML('')

            gr.HTML('<div class="sec-label">Retrieval settings</div>')
            doc_type_dd = gr.Dropdown(
                label="Document-type filter", choices=["All"], value="All",
                info="Restrict retrieval to a single document type.",
            )
            topk_slider = gr.Slider(
                label="Top-k chunks shown to the LLM",
                minimum=1, maximum=8, value=3, step=1,
                info="Higher = more context · lower = more focused.",
            )

            gr.HTML('<div class="sec-label">Index summary</div>')
            doc_info_md = gr.HTML(
                '<div class="src-empty">Upload and index a PDF to see '
                'its document-type breakdown here.</div>'
            )

        # ---- Right column (chat + sources) ---------------------------
        with gr.Column(scale=3, min_width=600):
            gr.HTML('<div class="sec-label">Conversation</div>')
            chatbot = gr.Chatbot(label="Conversation", show_label=False,
                                 height=540, elem_id="chat-area")

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask anything about the indexed document…",
                    label="", scale=4, container=False,
                )
                send_btn = gr.Button("Send", variant="primary",
                                     scale=0, min_width=110)

            with gr.Row():
                clear_btn  = gr.Button("Clear conversation",
                                       variant="secondary",
                                       scale=1, min_width=180)
                conf_label = gr.HTML('<div class="conf-empty">—</div>')

            with gr.Group(elem_classes=["examples-holder"]):
                gr.Examples(SAMPLE_PROMPTS, inputs=user_input,
                            label="Suggested queries")

            gr.HTML('<div class="sec-label">Sources for the last answer</div>')
            sources_md = gr.HTML('<div class="src-empty">'
                                 'Sources will appear here after your first question.'
                                 '</div>')

    # ---- Wiring ----
    process_btn.click(process_pdf, inputs=[pdf_input],
                      outputs=[engine_state, status_md, doc_info_md, doc_type_dd])

    chat_inputs  = [user_input, chatbot, engine_state, doc_type_dd, topk_slider]
    chat_outputs = [chatbot, sources_md, conf_label, user_input]
    send_btn.click(chat_handler, inputs=chat_inputs, outputs=chat_outputs)
    user_input.submit(chat_handler, inputs=chat_inputs, outputs=chat_outputs)
    clear_btn.click(clear_chat, outputs=[chatbot, sources_md, conf_label])


if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0")
