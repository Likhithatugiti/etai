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
        # Removed self._index, self._bm25, and self._embeddings 
        # to prevent storing heavy state in RAM.

    def build(self, chunks: List[Dict]):
        """
        Instead of building indexes here, treat this as a metadata store 
        only. Use an external Vector DB for the heavy lifting.
        """
        self._chunks = chunks

    def retrieve(self, query: str, entities: Dict[str, Any]) -> List[Dict]:
        if not self._chunks:
            return []

        # 1. Sparse Search (BM25)
        # If possible, offload this to your vector DB or use a very small 
        # in-memory dictionary. Do not keep heavy objects.
        from rank_bm25 import BM25Okapi
        texts = [c["text"] for c in self._chunks]
        tokenized_corpus = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        bm25_scores = bm25.get_scores(query.lower().split())
        
        # 2. Dense Search (API-based)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        q_emb = embedder.embed_query(query)
        # Instead of local FAISS search, if your data is large, 
        # this logic should be a call to a Vector DB API (Chroma/Pinecone).
        
        # 3. Knowledge Graph Traversal
        tags = EQUIPMENT_PATTERN.findall(query)
        kg_boost = np.zeros(len(self._chunks))
        if tags:
            seed_nodes = self.kg.find_equipment_nodes(tags)
            if seed_nodes:
                subgraph = self.kg.get_subgraph(seed_nodes, hops=2)
                node_texts = {n["id"] for n in subgraph["nodes"]}
                for i, chunk in enumerate(self._chunks):
                    for node_id in node_texts:
                        key = node_id.split("::")[-1]
                        if key and key in chunk["text"]:
                            kg_boost[i] += 0.5
                            break

        # Calculate hybrid scores without keeping massive FAISS objects
        final_scores = bm25_scores + kg_boost
        sorted_indices = np.argsort(final_scores)[::-1][:self.top_k]

        results = []
        for i in sorted_indices:
            c = dict(self._chunks[i])
            c["hybrid_score"] = float(final_scores[i])
            results.append(c)
            
        # Explicitly clear temporary objects
        del bm25
        del embedder
        return results
