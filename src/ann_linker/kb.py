"""
Adapted from:
- https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/ann_kb.py
- https://github.com/explosion/spaCy/blob/master/spacy/kb/kb_in_memory.pyx
"""

from typing import Iterable, TypedDict

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pydantic import create_model
from spacy.tokens import Span

from src.ann_linker.types import Alias, Entity

FAST_AND_SMALL = "sentence-transformers/all-MiniLM-L6-v2"


class EntityCandidate(TypedDict):
    alias: Alias
    _distance: float


class AnnKnowledgeBase:

    def __init__(self, uri: str):
        """Create an AnnKnowledgeBase."""
        # connect to the LanceDB
        self.db = lancedb.connect(uri)

        # Embedding model for aliases and entities
        # any LanceDB-compatible model is available as a drop-in replacement
        # ref: https://lancedb.github.io/lancedb/embeddings/default_embedding_functions/#text-embedding-functions
        self.encoder = (
            get_registry().get("sentence-transformers").create(name=FAST_AND_SMALL, device="cpu")
        )

        # we need pydantic classes to define the Arrow schemas of LanceDB tables.
        # those tables will contain the embedding from our encoder to do the ANN search.
        # because we want to use the local self.encoder, we can't use the traditional pydantic syntax.
        # instead, we create the pydantic classes dynamically using pydantic.create_model
        # ref: https://docs.pydantic.dev/latest/api/base_model/#pydantic.create_model
        self.LanceAlias = create_model(
            "LanceAlias",
            __base__=LanceModel,
            alias=(Alias, ...),
            vector=(Vector(self.encoder.ndims()), self.encoder.VectorField()),
        )
        self.LanceEntity = create_model(
            "LanceEntity",
            __base__=LanceModel,
            entity=(Entity, ...),
            vector=(Vector(self.encoder.ndims()), self.encoder.VectorField()),
        )

        # Parameters for the mention candidate generation
        self.top_k = 10
        self.max_distance = 0.5

        self._initialize_db()

    def _embed(self, text: str) -> list[int]:
        return self.encoder.generate_embeddings([text])[0]

    def _initialize_db(self):
        # TODO: do better, use mode from params: e.g. the table might already exists
        self.db.create_table("aliases", schema=self.LanceAlias, mode="overwrite")
        self.db.create_table("entities", schema=self.LanceEntity, mode="overwrite")

    def add_aliases(self, aliases: list[Alias]):
        """Build the ANN index of aliases in LanceDB."""
        table = self.db.open_table("aliases")
        table.add(
            [self.LanceAlias(alias=alias, vector=self._embed(alias.alias)) for alias in aliases]
        )

    def get_candidates_batch(self, mentions: Iterable[Span]) -> Iterable[Iterable[Alias]]:
        return [self.get_candidates(span) for span in mentions]

    def get_candidates(self, mention: Span) -> list[Alias]:
        return self.get_alias_candidates(mention.text)

    def get_alias_candidates(self, query: str) -> list[Alias]:
        """Embed a mention query, search ANN neighbours against the aliases index."""
        table = self.db.open_table("aliases")
        results = (
            table.search(self._embed(query))
            .metric("cosine")
            .limit(self.top_k)
            .select(["alias"])
            .to_list()
        )
        filtered_results = [r for r in results if r["_distance"] < self.max_distance]
        return [Alias(**result["alias"]) for result in filtered_results]

    def _aliases_to_entities(self, aliases: list[Alias]) -> list[str]:
        return list(set(entity_id for alias in aliases for entity_id in alias.entities))

    def get_entity_candidates(self, query: str) -> list[str]:
        """Get the entity IDs corresponding to a mention."""
        return self._aliases_to_entities(aliases=self.get_alias_candidates(query))

    def add_entities(self, entities: list[Entity]):
        """Build the ANN index of entities in LanceDB."""
        table = self.db.open_table("entities")
        table.add(
            [
                self.LanceEntity(entity=entity, vector=self._embed(entity.description))
                for entity in entities
            ]
        )

    def disambiguate(
        self, candidate_entities: list[str], doc_embedding: list[int]
    ) -> list[tuple[Entity, float]]:
        """Disambiguate candidate entities by getting the most similar to the context in the doc."""
        table = self.db.open_table("entities")
        entities_results = (
            # search the entity ANN index by the embedding of the context in the doc
            table.search(doc_embedding)
            .metric("cosine")
            # prefilter for only the candidate entities
            .where(f"list_has({candidate_entities}, entity.entity_id)", prefilter=True)
            # get the top_k
            .limit(self.top_k)
            # serialize
            .select(["entity"])
            .to_list()
        )
        return [(Entity(**result["entity"]), result["_distance"]) for result in entities_results]

    """
    We looked at the spacy abstractions closely:
        - spacy.kb.KnowledgeBase
        - spacy.kb.Candidate
    We decided to implement a class that does not use them.
    The reason is that we felt we were shoehorning too much of spacy classes.

    Still, the native spacy classes have the following useful methods that
    were not implemented here:
        - to_bytes
        - from_bytes
        - to_disk
        - from_disk
    """
