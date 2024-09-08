"""
Adapted from:
- https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/ann_kb.py
- https://github.com/explosion/spaCy/blob/master/spacy/kb/kb_in_memory.pyx
"""

from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase
from spacy.vocab import Vocab


class AnnKnowledgeBase(KnowledgeBase):

    def __init__(self, vocab: Vocab, entity_vector_length: int):
        """Create an AnnKnowledgeBase."""
        super().__init__(vocab, entity_vector_length)

    ### CORE
    # add_entity
    # add_alias
    # get_candidates
    #   -> get_alias_candidates
    # get_vector

    ### NIT
    # get_alias_strings
    # get_prior_prob

    ### I/O
    # to_bytes
    # from_bytes
    # to_disk
    # from_disk
