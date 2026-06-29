# ⚙️ AXON — Industrial Knowledge Intelligence

> **AI for Industrial Knowledge Intelligence: Unified Asset & Operations Brain**  
> ET AI Hackathon 2026 — Problem Statement 8

---

## What is AXON?

AXON is a **multi-agent reasoning system** for industrial document intelligence. Unlike systems that route a query to a single agent and return one answer, AXON runs a full reasoning loop:

1. **Planner** decomposes the query into sub-questions and identifies intent (maintenance / compliance / lessons / knowledge)
2. **Hybrid Retriever** fetches context via BM25 sparse search + FAISS dense search + Knowledge Graph traversal — then CrossEncoder re-ranks
3. **Specialist Agents** (Knowledge, Maintenance, Compliance, Lessons Learned) answer in parallel
4. **Critic** scores each answer on confidence — low confidence loops back to retrieval
5. **Synthesizer** merges all agent answers into one coherent response with citations
6. **Hallucination Guard** verifies final answer against retrieved chunks
7. **Engineer Feedback** corrects the knowledge base — both the KG and vector index update live

---

## Architecture

```
USER QUERY
    │
    ▼
┌──────────┐     decomposes into sub-questions + intent tags
│ Planner  │
└────┬─────┘
     │
     ▼
┌──────────────────────────────────┐
│         Hybrid Retriever         │
│  BM25 + FAISS + KG Traversal     │
│  → CrossEncoder Re-rank          │
└────────────────┬─────────────────┘
                 │ (parallel fan-out based on intent_tags)
    ┌────────────┼──────────────┬──────────────┐
    ▼            ▼              ▼              ▼
Knowledge    Maintenance    Compliance    Lessons
 Agent         Agent          Agent         Agent
    └────────────┴──────────────┴──────────────┘
                        │
                        ▼
                   ┌─────────┐
                   │  Critic  │  ← scores confidence
                   └────┬────┘
           ┌────────────┴────────────┐
        retry                     proceed
        (retriever)            (synthesizer)
                                     │
                                     ▼
                           ┌──────────────────┐
                           │   Synthesizer     │
                           └────────┬─────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Hallucination Guard  │
                         └──────────┬────────────┘
                             flag ──┤── clear → END
                          (re-synthesize)

Engineer Feedback → Update Agent → KG + FAISS delta update
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph StateGraph |
| LLM | Groq (llama3-70b) / Gemini 1.5 Flash (switchable) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Sparse retrieval | BM25 (rank-bm25) |
| Dense retrieval | FAISS |
| Re-ranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Knowledge Graph | NetworkX with typed edges |
| NLP / NER | spaCy |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| Doc parsing | pdfplumber, python-docx, python-pptx, pandas |

---

## Project Structure

```
axon/
├── agents/
│   ├── planner.py           # Query decomposition + intent tagging
│   ├── retriever_node.py    # Hybrid retrieval node
│   ├── knowledge_agent.py   # General engineering queries
│   ├── maintenance_agent.py # RCA + FMEA + maintenance recommendations
│   ├── compliance_agent.py  # Regulatory gap detection (OISD, PESO, BIS)
│   ├── lessons_agent.py     # Failure intelligence + incident patterns
│   ├── critic.py            # Confidence scoring + retry logic
│   ├── synthesizer.py       # Multi-agent answer merger
│   ├── hallucination_guard.py # Factual verification
│   ├── ingestion_agent.py   # Document ingestion node
│   ├── update_agent.py      # Engineer feedback → KG + index update
│   └── orchestrator.py      # Intent routing
├── graph/
│   └── knowledge_graph.py   # Typed-edge NetworkX KG
├── retrieval/
│   └── hybrid_retriever.py  # BM25 + FAISS + KG + CrossEncoder
├── ingestion/
│   └── document_ingestor.py # PDF, DOCX, PPTX, CSV, XLSX, TXT parser
├── api/
│   ├── routes.py            # FastAPI endpoints
│   └── context.py           # Singleton KG/retriever/pipeline holder
├── prompts/
│   └── system_prompts.py    # All LLM prompts
├── utils/
│   └── llm.py               # LLM factory (Groq / Gemini)
├── pipeline.py              # LangGraph graph assembly
├── main.py                  # FastAPI entrypoint
├── app.py                   # Streamlit UI
├── state.py                 # AgentState TypedDict
├── requirements.txt
├── .env.template
└── .gitignore
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Likhithatugiti/Industrial_intelligence.git
cd Industrial_intelligence
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Set up environment
cp .env.template .env
# Edit .env — add your GROQ_API_KEY or GOOGLE_API_KEY

# 3. Start the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Launch the UI (separate terminal)
streamlit run app.py
```

API docs available at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest` | Upload documents (PDF/DOCX/PPTX/CSV/XLSX/TXT) |
| POST | `/query` | Run the AXON reasoning pipeline |
| POST | `/feedback` | Submit engineer correction → live KG + index update |
| GET | `/kg/stats` | Knowledge graph node/edge statistics |
| GET | `/health` | Liveness check |

---

## What makes AXON different from single-pass RAG systems

| Capability | Single-pass RAG | AXON |
|---|---|---|
| Query decomposition | ❌ | ✅ Planner agent |
| Multi-agent parallel reasoning | ❌ | ✅ 4 specialist agents |
| Self-correction loop | ❌ | ✅ Critic → retry on low confidence |
| Hallucination detection | ❌ | ✅ Hallucination guard node |
| Sparse + dense + graph retrieval | ❌ dense only | ✅ BM25 + FAISS + KG traversal |
| Live knowledge update | ❌ | ✅ Engineer feedback → delta update |
| Typed knowledge graph | ❌ | ✅ Equipment / Regulation / FailureMode nodes |
| Equipment tag extraction (regex) | ❌ | ✅ P-101, V-201 pattern matching |
| Regulation reference detection | ❌ | ✅ OISD, PESO, BIS, Factory Act |
