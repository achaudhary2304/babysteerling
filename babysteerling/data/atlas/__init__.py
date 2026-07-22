"""Atlas-inspired concept dataset construction pipeline (4 stages, each idempotent):

    tag_chunks -> build_concepts -> assign_concepts -> tokenize_dataset

Independently implements, at laptop scale, the ideas described in Guide Labs' Atlas pipeline
(see the project's NOTICE for attribution): LLM-tag a text corpus, cluster the tags into a
canonical concept library, assign concepts back to the corpus, then tokenize for training.
"""
from .assign_concepts import assign_concepts
from .build_concepts import build_concepts
from .tag_chunks import tag_chunks
from .tokenize_dataset import tokenize_dataset

__all__ = ["tag_chunks", "build_concepts", "assign_concepts", "tokenize_dataset"]
