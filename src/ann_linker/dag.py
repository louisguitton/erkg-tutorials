import spacy
import srsly
from spacy.language import Language

from src.ann_linker.kb import AnnKnowledgeBase
from src.ann_linker.types import Alias, Entity


def entities() -> list[Entity]:
    return [Entity(**entity) for entity in srsly.read_jsonl("data/test-microsoft/entities.jsonl")]


def aliases() -> list[Alias]:
    return [Alias(**alias) for alias in srsly.read_jsonl("data/test-microsoft/aliases.jsonl")]


def nlp() -> Language:
    return spacy.load("en_core_web_md")


def kb(entities: list[Entity], aliases: list[Alias]) -> AnnKnowledgeBase:
    uri = "data/sample-lancedb"
    ann_kb = AnnKnowledgeBase(uri=uri)
    ann_kb.add_entities(entities)
    ann_kb.add_aliases(aliases)
    return ann_kb


def ann_linker(nlp, kb, cg):
    """Adapted from https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/cli/create_index.py"""
    ann_linker = nlp.create_pipe("ann_linker")
    ann_linker.set_kb(kb)
    ann_linker.set_cg(get_candidates)  # get_candidates_batch
    return ann_linker


def nlp_with_linker(nlp, ann_linker):
    """Adapted from https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/cli/create_index.py"""
    nlp.add_pipe(ann_linker, last=True)

    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    nlp.from_disk(output_dir)
