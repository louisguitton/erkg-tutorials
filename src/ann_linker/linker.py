from pathlib import Path
from typing import Tuple

from spacy.language import Language
from spacy.tokens import Doc, Span

from src.ann_linker.kb import AnnKnowledgeBase


class AnnLinker:
    """The AnnLinker adds Entity Linking capabilities
    to map NER mentions to KnowledgeBase Aliases or directly to KnowledgeBase Ids
    """

    def __init__(self, **cfg):
        """Initialize the AnnLinker

        nlp (Language): spaCy Language object
        """
        Span.set_extension("alias_candidates", default=[], force=True)
        Span.set_extension("kb_candidates", default=[], force=True)

        self.kb = None
        self.use_disambiguation_threshold = False

    def __call__(self, doc: Doc) -> Doc:
        """Annotate spaCy doc.ents with candidate info.
        If disambiguate is True, use entity vectors and doc context
        to pick the most likely Candidate

        doc (Doc): spaCy Doc

        RETURNS (Doc): spaCy Doc with updated annotations
        """
        self.require_kb()

        mentions = doc.ents
        batch_candidates = self.kb.get_candidates_batch(mentions)

        for ent, alias_candidates in zip(doc.ents, batch_candidates):
            ent._.alias_candidates = alias_candidates

            if len(alias_candidates) == 0:
                continue
            else:
                candidate_entities = self.kb._aliases_to_entities(alias_candidates)

                # TODO: have a configurable context (e.g. -1/+1 sentence)
                context_embedding = self.kb._embed(ent.sent)

                kb_candidates = self.kb.disambiguate(
                    candidate_entities, context_embedding, ent.text
                )

                ent._.kb_candidates = kb_candidates

                if self.use_disambiguation_threshold:
                    filtered_results = [
                        (entity, cosine_score)
                        for entity, cosine_score in kb_candidates
                        if cosine_score < self.kb.max_distance
                    ]
                else:
                    filtered_results = kb_candidates

                if len(filtered_results):
                    best_candidate = filtered_results[0][0]
                    for token in ent:
                        token.ent_kb_id_ = best_candidate.entity_id
        return doc

    def set_kb(self, kb: AnnKnowledgeBase):
        """Set the KnowledgeBase."""
        self.kb = kb

    def require_kb(self):
        """Raise an error if the kb is not set.

        RAISES:
            ValueError: kb required
        """
        if getattr(self, "kb", None) in (None, True, False):
            raise ValueError(f"KnowledgeBase `kb` required for {self.name}")

    def from_disk(self, path: Path, **kwargs):
        """Deserialize saved AnnLinker from disk."""
        raise NotImplementedError("This is not available at this time.")

    def to_disk(self, path: Path, exclude: Tuple = tuple(), **kwargs):
        """Serialize AnnLinker to disk."""
        raise NotImplementedError("This is not available at this time.")


@Language.factory("ann_linker")
def create_ann_linker(nlp: Language, name: str):
    return AnnLinker()
