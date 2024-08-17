import spacy
from gliner_spacy.pipeline import (  # noqa: F401 because we need to register the factory with spacy
    GlinerSpacy,
)

candidate_labels = [
    "persons",
    "address",
    "shell companies",
    "banks or law firms",
]  # NuZero requires labels to be lower-cased

model_name = "numind/NuZero_token"

nlp = spacy.load("en_core_web_md", disable=["ner"])
# nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-mbert-base-multinerd"})
nlp.add_pipe(
    "gliner_spacy",
    config={
        "gliner_model": model_name,
        "chunk_size": 250,
        "labels": candidate_labels,
        "style": "ent",
        "threshold": 0.3,
    },
)
