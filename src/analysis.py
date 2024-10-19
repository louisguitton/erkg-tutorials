from typing import Iterator

import pandas as pd
from spacy.tokens import Doc


def analyse_el_docs(docs: Iterator[Doc]) -> list[pd.DataFrame]:
    for_review: list[pd.DataFrame] = []
    for doc in docs:
        records = []
        for phrase in doc._.phrases[:30]:
            records.append(
                (
                    phrase.text,
                    phrase.rank,
                    phrase.count,
                    [
                        {"text": text, "kb_id": kb_id}
                        for text, kb_id in set(
                            (ent.text, ent.kb_id_) for chunk in phrase.chunks for ent in chunk.ents
                        )
                    ],
                )
            )
        raw_entities = pd.DataFrame.from_records(
            records, columns=["phrase", "rank", "count", "entities"]
        ).explode("entities")
        df = pd.concat(  # type: ignore
            [
                raw_entities.drop(columns="entities"),
                pd.json_normalize(raw_entities.entities).set_index(raw_entities.index),  # type: ignore
            ],
            axis=1,
        )
        entities_to_review = df.loc[lambda d: (d.text.notnull()) & (d.kb_id == "")]
        for_review.append(entities_to_review)
    return for_review
