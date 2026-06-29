"""
Three-stage Hybrid Retrieval
1. Sparse  – BM25 (rank_bm25)        → catches exact equipment tags / reg refs
2. Dense   – FAISS + sentence-transformers → semantic similarity
3. Graph   – KG 2-hop neighbourhood  → relational context
4. Re-rank – CrossEncoder             → final top-k
"""

from __future__ import annotations
import re
import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.knowledge_graph import IndustrialKG

EQUIPMENT_PATTERN = re.compile(r"\b[A-Z]{1,3}-\d{2,4}[A-Z]?\b")


class HybridRetriever:
    def __init__(self, kg: "IndustrialKG", top_k: int = 8):
        self.kg = kg
        self.top_k = top_k
        self._chunks: List[Dict] = []
        self._index = None          # FAISS index
        self._embedder = None
        self._bm25 = None
        self._reranker = None
        self._tokenized_corpus: List[List[str]] = []

    # ── Build index ─────────────────────────────────────────────────────────

    def build(self, chunks: List[Dict]):
        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        # BM25
        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # FAISS + embeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        import faiss
        import os
        
        # We use Google Gemini for embeddings to save local memory
        self._embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        embeddings_list = self._embedder.embed_documents(texts)
        embeddings = np.array(embeddings_list, dtype=np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)
        self._embeddings = embeddings

        self._reranker = None

    def update(self, new_chunks: List[Dict]):
        """Incremental update — rebuild on new chunks only."""
        all_chunks = self._chunks + new_chunks
        self.build(all_chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, entities: Dict[str, Any]) -> List[Dict]:
        if not self._chunks:
            return []

        candidate_ids: Dict[int, float] = {}

        # 1. BM25 sparse
        bm25_scores = self._bm25.get_scores(query.lower().split())
        top_bm25 = np.argsort(bm25_scores)[::-1][:self.top_k * 2]
        for idx in top_bm25:
            candidate_ids[int(idx)] = candidate_ids.get(int(idx), 0) + float(bm25_scores[idx])

        # 2. FAISS dense
        import faiss
        q_emb_list = self._embedder.embed_query(query)
        q_emb = np.array([q_emb_list], dtype=np.float32)
        distances, indices = self._index.search(q_emb, self.top_k * 2)
        max_dist = float(distances[0].max()) + 1e-9
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                dense_score = 1.0 - (float(dist) / max_dist)
                candidate_ids[int(idx)] = candidate_ids.get(int(idx), 0) + dense_score

        # 3. KG graph traversal — seed with equipment tags in query
        tags = EQUIPMENT_PATTERN.findall(query)
        if tags:
            seed_nodes = self.kg.find_equipment_nodes(tags)
            if seed_nodes:
                subgraph = self.kg.get_subgraph(seed_nodes, hops=2)
                # find chunks that mention any node from the subgraph
                node_texts = {n["id"] for n in subgraph["nodes"]}
                for i, chunk in enumerate(self._chunks):
                    for node_id in node_texts:
                        key = node_id.split("::")[-1]
                        if key and key in chunk["text"]:
                            candidate_ids[i] = candidate_ids.get(i, 0) + 0.5
                            break

        # collect candidates
        sorted_ids = sorted(candidate_ids, key=lambda i: candidate_ids[i], reverse=True)
        candidates = [self._chunks[i] for i in sorted_ids[:self.top_k * 2] if i < len(self._chunks)]

        # 4. CrossEncoder re-rank (Removed to save memory)
        # We rely entirely on the hybrid score

        # fallback: return by hybrid score
        results = []
        for i in sorted_ids[:self.top_k]:
            if i < len(self._chunks):
                c = dict(self._chunks[i])
                c["hybrid_score"] = float(candidate_ids[i])
                c["reranker_score"] = 0.0
                results.append(c)
        return results
