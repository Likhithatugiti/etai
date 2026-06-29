"""
Application-level singletons — KG, retriever, compiled pipeline.
Initialised once at startup in main.py.
"""

from graph import IndustrialKG
from retrieval import HybridRetriever

_kg: IndustrialKG = None
_retriever: HybridRetriever = None
_pipeline = None


def init_context(kg: IndustrialKG, retriever: HybridRetriever, pipeline):
    global _kg, _retriever, _pipeline
    _kg = kg
    _retriever = retriever
    _pipeline = pipeline


def get_kg() -> IndustrialKG:
    return _kg


def get_retriever() -> HybridRetriever:
    return _retriever


def get_pipeline():
    return _pipeline
