"""
Typed-edge Knowledge Graph built on NetworkX.

Node types : Equipment | Procedure | Regulation | FailureMode | Document | Person
Edge types  : HAS_FAILURE_MODE | CAUSED_BY | GOVERNED_BY | REFERENCES |
              DOCUMENTED_IN | CONTAINS | PART_OF | REPORTED_BY
"""

import networkx as nx
import spacy
import re
from typing import Dict, List, Any, Optional

# Industrial entity patterns (equipment tags, reg refs)
EQUIPMENT_PATTERN = re.compile(r"\b[A-Z]{1,3}-\d{2,4}[A-Z]?\b")  # P-101, V-201A
REGULATION_PATTERN = re.compile(
    r"\b(OISD[-\s]\d+|PESO[-\s]\w+|IS[-\s]\d+|Factory Act|BIS[-\s]\w+|ISO[-\s]\d+)\b",
    re.IGNORECASE,
)


class IndustrialKG:
    def __init__(self):
        self.graph = nx.DiGraph()
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            self._nlp = None

    # ── Node helpers ────────────────────────────────────────────────────────

    def _add_node(self, node_id: str, node_type: str, **attrs):
        self.graph.add_node(node_id, type=node_type, **attrs)

    def _add_edge(self, src: str, dst: str, rel: str, **attrs):
        self.graph.add_edge(src, dst, relation=rel, **attrs)

    # ── Ingestion ───────────────────────────────────────────────────────────

    def add_chunk(self, chunk: Dict[str, Any]):
        """Extract entities from a chunk and wire them into the KG."""
        doc_id = chunk.get("doc_id", chunk.get("document_name", "unknown"))
        chunk_id = chunk.get("chunk_id", id(chunk))
        text = chunk.get("text", "")

        # Document node
        doc_node = f"doc::{doc_id}"
        self._add_node(doc_node, "Document", title=doc_id,
                       doc_type=chunk.get("document_type", "unknown"))

        # Chunk node
        chunk_node = f"chunk::{chunk_id}"
        self._add_node(chunk_node, "Chunk", text=text[:300],
                       page=chunk.get("page", 0), doc_id=doc_id)
        self._add_edge(doc_node, chunk_node, "CONTAINS")

        # Equipment tags
        for tag in EQUIPMENT_PATTERN.findall(text):
            eq_node = f"equipment::{tag}"
            self._add_node(eq_node, "Equipment", tag=tag)
            self._add_edge(chunk_node, eq_node, "REFERENCES")

        # Regulation references
        for ref in REGULATION_PATTERN.findall(text):
            reg_node = f"reg::{ref.strip()}"
            self._add_node(reg_node, "Regulation", ref=ref.strip())
            self._add_edge(chunk_node, reg_node, "GOVERNED_BY")
            self._add_edge(eq_node if EQUIPMENT_PATTERN.search(text) else chunk_node,
                           reg_node, "GOVERNED_BY")

        # NLP entities (ORG, PRODUCT, PERSON)
        if self._nlp:
            doc = self._nlp(text[:512])
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT", "PERSON", "LAW"):
                    ent_node = f"entity::{ent.label_}::{ent.text}"
                    self._add_node(ent_node, ent.label_, name=ent.text)
                    self._add_edge(chunk_node, ent_node, "MENTIONS")

    def add_failure_link(self, equipment_tag: str, failure_mode: str,
                         cause: Optional[str] = None):
        """Manually wire a failure mode for an equipment tag."""
        eq_node = f"equipment::{equipment_tag}"
        fm_node = f"failure::{failure_mode}"
        self._add_node(eq_node, "Equipment", tag=equipment_tag)
        self._add_node(fm_node, "FailureMode", mode=failure_mode)
        self._add_edge(eq_node, fm_node, "HAS_FAILURE_MODE")
        if cause:
            cause_node = f"failure::{cause}"
            self._add_node(cause_node, "FailureMode", mode=cause)
            self._add_edge(fm_node, cause_node, "CAUSED_BY")

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_subgraph(self, node_ids: List[str], hops: int = 2) -> Dict:
        """Return nodes+edges within `hops` of given seed nodes."""
        visited = set()
        queue = list(node_ids)
        for _ in range(hops):
            next_q = []
            for n in queue:
                if n not in self.graph:
                    continue
                for nbr in list(self.graph.successors(n)) + list(self.graph.predecessors(n)):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_q.append(nbr)
            queue = next_q
        seed_set = set(node_ids) | visited
        sub = self.graph.subgraph(seed_set)
        return {
            "nodes": [{"id": n, **self.graph.nodes[n]} for n in sub.nodes],
            "edges": [
                {"src": u, "dst": v, "relation": d.get("relation")}
                for u, v, d in sub.edges(data=True)
            ],
        }

    def find_equipment_nodes(self, tags: List[str]) -> List[str]:
        return [f"equipment::{t}" for t in tags if f"equipment::{t}" in self.graph]

    def stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(
                __import__("collections").Counter(
                    d.get("type", "?") for _, d in self.graph.nodes(data=True)
                )
            ),
        }

    # ── Update (feedback loop) ───────────────────────────────────────────────

    def update_from_correction(self, correction_text: str, doc_id: str = "engineer_feedback"):
        """Ingest engineer correction text as a new chunk."""
        chunk = {
            "doc_id": doc_id,
            "chunk_id": f"feedback_{id(correction_text)}",
            "text": correction_text,
            "document_type": "engineer_feedback",
            "page": 0,
        }
        self.add_chunk(chunk)
