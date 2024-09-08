"""
Adapted from:
- https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/ann_kb.py
- https://github.com/explosion/spaCy/blob/master/spacy/kb/kb_in_memory.pyx
"""

# ref: https://lancedb.github.io/lancedb/embeddings/available_embedding_models/text_embedding_functions/sentence_transformers/
from typing import Iterable

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase
from spacy.tokens import Span
from spacy.vocab import Vocab


class AnnKnowledgeBase(KnowledgeBase):

    def __init__(self, vocab: Vocab, entity_vector_length: int):
        """Create an AnnKnowledgeBase."""
        super().__init__(vocab, entity_vector_length)

    ### CORE
    # add_entity
    # add_alias
    def get_candidates(self, mention: Span) -> Iterable[Candidate]:
        return self.get_alias_candidates(mention.text)

    def get_alias_candidates(self, alias: str) -> Iterable[Candidate]:
        # vectorize -> search ANN neighbours against aliases index -> build Iterable[Candidate]
        return []

    def get_vector(self, entity: str):
        pass

    ### NIT
    # get_alias_strings
    # get_prior_prob

    ### I/O
    # to_bytes
    # from_bytes
    # to_disk
    # from_disk
    # from_disk
