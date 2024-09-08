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


def nlp_with_linker(nlp: Language, kb: AnnKnowledgeBase) -> Language:
    ann_linker = nlp.add_pipe("ann_linker", last=True)
    ann_linker.set_kb(kb)
    return nlp
