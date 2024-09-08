from typing import Optional

import spacy
import srsly
from pydantic import BaseModel
from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase
from spacy.language import Language

INPUT_DIM = 300  # dimension of pretrained input vectors


class Entity(BaseModel):
    entity_id: str
    name: str
    description: str
    label: Optional[str] = None


class Alias(BaseModel):
    alias: str
    entities: list[str]
    probabilities: list[float]


def entities() -> list[Entity]:
    return [Entity(**entity) for entity in srsly.read_jsonl("data/test-microsoft/entities.jsonl")]


def aliases() -> list[Alias]:
    return [Alias(**alias) for alias in srsly.read_jsonl("data/test-microsoft/aliases.jsonl")]


def nlp() -> Language:
    return spacy.load("en_core_web_md")


def kb(nlp: Language, entities: list[Entity], aliases: list[Alias]) -> KnowledgeBase:
    """Adapted from https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/cli/create_index.py"""

    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=INPUT_DIM)

    # set up the data
    entity_ids = []
    descriptions = []
    freqs = []
    for e in entities:
        entity_ids.append(e.entity_id)
        descriptions.append(e.description)
        freqs.append(100)

    # get the pretrained entity vectors
    embeddings = [nlp.make_doc(desc).vector for desc in descriptions]

    # set the entities, can also be done by calling `kb.add_entity` for each entity
    for i in range(len(entity_ids)):
        entity = entity_ids[i]
        if not kb.contains_entity(entity):
            kb.add_entity(entity, freqs[i], embeddings[i])

    for a in aliases:
        ents = [e for e in a.entities if kb.contains_entity(e)]
        n_ents = len(ents)
        if n_ents > 0:
            prior_prob = [1.0 / n_ents] * n_ents
            kb.add_alias(alias=a.alias, entities=ents, probabilities=prior_prob)

    return kb


def cg():
    """Adapted from https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/cli/create_index.py"""
    cg = CandidateGenerator().fit(kb.get_alias_strings(), verbose=True)
    return cg


def ann_linker(nlp, kb, cg):
    """Adapted from https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/cli/create_index.py"""
    ann_linker = nlp.create_pipe("ann_linker")
    ann_linker.set_kb(kb)
    ann_linker.set_cg(cg)
    return ann_linker


def nlp_with_linker(nlp, ann_linker):
    """Adapted from https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/cli/create_index.py"""
    nlp.add_pipe(ann_linker, last=True)

    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    nlp.from_disk(output_dir)
