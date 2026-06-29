"""
AXON Streamlit UI
Connects to the FastAPI backend at localhost:8000.
Run: streamlit run app.py
(Start the FastAPI server first: uvicorn main:app --port 8000)
"""

import streamlit as st
import requests
import json

API = "https://etai-ybmt.onrender.com/"

st.set_page_config(page_title="AXON — Industrial Knowledge Intelligence", layout="wide")

st.title("⚙️ AXON — Industrial Knowledge Intelligence")
st.caption("Unified Asset & Operations Brain | Multi-Agent RAG + LangGraph + KG Traversal")

# ── Sidebar: KG stats ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Knowledge Graph")
    try:
        # Changed "/kg/stats" to "kg/stats"
        stats = requests.get(f"{API.rstrip('/')}/kg/stats", timeout=3).json()
        
        st.metric("Nodes", stats.get("nodes", 0))
        st.metric("Edges", stats.get("edges", 0))
        with st.expander("Node types"):
            st.json(stats.get("node_types", {}))
    except Exception as e:
        st.warning(f"Backend not reachable: {e}")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_ingest, tab_query, tab_feedback = st.tabs(
    ["📂 Ingest Documents", "🔍 Query", "🔄 Engineer Feedback"]
)

# ── INGEST ────────────────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Upload Industrial Documents")
    files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "pptx", "csv", "xlsx", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not files):
        with st.spinner("Ingesting..."):
            resp = requests.post(
                f"{API}/ingest",
                files=[("files", (f.name, f.read(), f.type)) for f in files],
            )
        if resp.ok:
            data = resp.json()
            st.success(
                f"Ingested **{data['files_ingested']}** file(s), "
                f"**{data['chunks_created']}** chunks created."
            )
            st.json(data["kg_stats"])
        else:
            st.error(f"Ingestion failed: {resp.text}")

# ── QUERY ─────────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask a Question")
    query = st.text_input("Enter your query", placeholder="What caused the P-101 seal failure last quarter?")

    if st.button("Run AXON", disabled=not query):
        with st.spinner("Running AXON reasoning pipeline..."):
            resp = requests.post(
                f"{API}/query",
                json={"query": query},
                timeout=120,
            )

        if resp.ok:
            data = resp.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Intent Tags", ", ".join(data.get("intent_tags", [])))
            col2.metric("Critic Score (avg)",
                        f"{sum(data.get('critic_scores', [0])) / max(len(data.get('critic_scores', [1])), 1):.2f}")
            col3.metric("Hallucination", "⚠️ Yes" if data.get("hallucination_flag") else "✅ Clear")

            st.subheader("Sub-questions decomposed by Planner")
            for i, sq in enumerate(data.get("sub_questions", []), 1):
                st.markdown(f"{i}. {sq}")

            st.subheader("Final Answer")
            st.markdown(data.get("final_answer", ""))

            with st.expander("Agent Answers (detailed)"):
                for item in data.get("agent_answers", []):
                    st.markdown(
                        f"**[{item['agent'].upper()}]** confidence: `{item.get('confidence', 0):.2f}`"
                    )
                    st.markdown(item.get("answer", ""))
                    st.divider()

            with st.expander("Source Citations"):
                for c in data.get("source_citations", []):
                    st.markdown(
                        f"- `{c['chunk_id']}` — **{c['document']}** p.{c['page']} (via {c['agent']})"
                    )

            if data.get("hallucination_flag"):
                st.warning(f"⚠️ Hallucination detected: {data.get('hallucination_reason')}")
        else:
            st.error(f"Query failed: {resp.text}")

# ── FEEDBACK ──────────────────────────────────────────────────────────────────
with tab_feedback:
    st.subheader("Engineer Knowledge Correction")
    st.caption(
        "Provide corrected or updated knowledge. "
        "It will be ingested into the KG and vector store immediately."
    )
    correction = st.text_area("Corrected knowledge", height=200,
                              placeholder="The P-101 pump seal was replaced on 2024-03-15 due to cavitation, not bearing failure as previously recorded.")
    if st.button("Submit Correction", disabled=not correction):
        with st.spinner("Updating knowledge base..."):
            resp = requests.post(
                f"{API}/feedback",
                json={"correction_text": correction},
                timeout=30,
            )
        if resp.ok:
            data = resp.json()
            st.success(
                f"Knowledge base updated! "
                f"{data['new_chunks']} new chunk(s) added. "
                f"KG now has {data['kg_stats']['nodes']} nodes."
            )
        else:
            st.error(f"Update failed: {resp.text}")
