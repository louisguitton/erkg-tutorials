import spacy
from spacy.language import Language
from zshot import PipelineConfig, displacy
from zshot.knowledge_extractor import KnowGL
from zshot.linker import Linker, LinkerRegen, LinkerSMXM
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.relation_extractor import RelationsExtractorZSRC
from zshot.utils.data_models import Entity, Relation

config = PipelineConfig(
    mentions_extractor=MentionsExtractorSpacy(),
    entities=[
        Entity(name="PERSON", description="People, including fictional"),
        Entity(name="WORK OF ART", description="Titles of books, songs, etc."),
        Entity(
            name="ORGANIZATION",
            description="Companies, agencies, institutions, organizations, etc.",
        ),
    ],
    linker=LinkerRegen(),
    #  linker=LinkerSMXM(),
    relations=[
        Relation(
            name="takes place in fictional universe",
            description="the subject is a work describing a fictional universe, i.e. whose plot occurs in this universe",
        ),  # https://www.wikidata.org/wiki/Property:P1434
        Relation(
            name="present in work",
            description="this (fictional or fictionalized) entity, place, or person appears in that work as part of the narration",
        ),  # https://www.wikidata.org/wiki/Property:P1441
        Relation(
            name="performer",
            description="actor, musician, band or other performer associated with this role or musical work",
        ),  # https://www.wikidata.org/wiki/Property:P175
        Relation(
            name="director",
            description="director(s) of film, TV-series, stageplay, video game or similar",
        ),  # https://www.wikidata.org/wiki/Property:P57
        Relation(
            name="followed by",
            description="immediately following item in a series of which the subject is a part",
        ),  # https://www.wikidata.org/wiki/Property:P156
    ],
    relations_extractor=RelationsExtractorZSRC(thr=0.8),
)

nlp = spacy.load("en_core_web_md", disable=["ner"])
nlp.add_pipe("zshot", config=config, last=True)
