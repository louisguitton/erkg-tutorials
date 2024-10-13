"""Extract Entity Linking artefacts from Senzing results.

Requires this data asset as input:
- https://storage.googleapis.com/erkg/icij/ICIJ-entity-report-2024-06-21_12-04-57-std.json.zip
"""

import csv
import json
import pathlib
import re
from collections import Counter
from enum import Enum
from typing import TypedDict

import pandas as pd
import spacy
from loguru import logger
from spacy.pipeline import EntityRuler
from spacy.tokens import DocBin
from tqdm import tqdm

from src.scraper import SPACY_MODEL


def load_countries(country_codes_path: str | pathlib.Path = "data/senzing/country.tsv") -> dict:
    """Map from a country code to a full name."""
    COUNTRIES: dict = {}

    logger.info(f"Loading country codes from {country_codes_path}")
    with open(country_codes_path, "r", encoding="utf-8") as fp:
        tsv_reader = csv.reader(fp, delimiter="\t")
        next(tsv_reader, None)  # skip the header row
        COUNTRIES = {row[0]: row[1] for row in tsv_reader}

    return COUNTRIES


def get_country(countries: dict, code: str | None) -> str | None:
    if code is None:
        return None
    return countries.get(code.strip())


class EntityFeature(Enum):
    ADDRESS = "ADDRESS"
    COUNTRY_OF_ASSOCIATION = "COUNTRY_OF_ASSOCIATION"
    DOB = "DOB"
    DUNS_NUMBER = "DUNS_NUMBER"
    GROUP_ASSOCIATION = "GROUP_ASSOCIATION"
    NAME = "NAME"
    PHONE = "PHONE"
    RECORD_TYPE = "RECORD_TYPE"
    REL_ANCHOR = "REL_ANCHOR"
    REL_POINTER = "REL_POINTER"
    WEBSITE = "WEBSITE"


def load_entities(
    icij_path: str | pathlib.Path = "data/ICIJ-entity-report-2024-06-21_12-04-57-std.json",
) -> dict[str, dict[EntityFeature, str]]:
    """Map from entity_id to the available entity features in the Senzing results."""
    ents: dict[str, dict[EntityFeature, str]] = {}

    logger.info(f"Parsing Senzing results: {icij_path}")
    num_lines = sum(1 for _ in open(icij_path))

    with tqdm(total=num_lines) as pbar:
        with open(icij_path, "r", encoding="utf-8") as fp:
            while line := fp.readline():
                dat = json.loads(line.strip())
                ent: dict = dat["RESOLVED_ENTITY"]

                ent_id: str = ent["ENTITY_ID"]

                features: dict[EntityFeature, str] = {
                    EntityFeature(key): feature[0]["FEAT_DESC"]
                    for key, feature in ent["FEATURES"].items()
                }

                ents[ent_id] = features

                pbar.update(1)

    return ents


PAT_LIST: list[str] = [
    r"^\-?(to\s+)?([the]+\s+)?bearer\.?\s?(\d+)?(\w)?$",
    r"^.*bearer.*shares?$",
    r"^the\s+bearer\s+\([\d\,]+\)$",
    r"^[ae]l\s+portador$",
    r"^the\s?bearer$",
    r"^bearer\s?warrant$",
    r"^bearer\s?shareholder$",
    r"^the\,\s+bearer$",
    r"^bearer\s+\(reedeem\s+shares\)$",
    r"^the\s+bearer\s+\(lost\)$",
    r"^bearer\s+\-\s+[\w]$",
    r"^bearer\s+\"\w\"$",
    r"^bearer\s+[\d\-]+$",
    r"^bearer\s+no\.\s+\d+$",
    r"^the\s+bearer\s+at\s+[\d\,]+$",
    r"^nan$",
    r"^[\?]+$",
]


def filter_bearer(name: str) -> bool:
    """These names are used to hide the identity of a company shareholder."""
    name = str(name).lower()

    for pat in PAT_LIST:
        if re.search(pat, name) is not None:
            return False

    return True


class EntityData(TypedDict):
    entity_id: str
    type: str
    name: str
    description: str


def get_entity_type(entity_features: dict[EntityFeature, str]) -> str:
    if EntityFeature.RECORD_TYPE.value in entity_features:
        if entity_features["RECORD_TYPE"] == "PERSON":
            return "PER"
        elif entity_features["RECORD_TYPE"] == "ORGANIZATION":
            return "ORG"
    if (
        EntityFeature.DOB.value in entity_features
        or EntityFeature.GROUP_ASSOCIATION.value in entity_features
    ):
        return "PER"
    if EntityFeature.DUNS_NUMBER.value or EntityFeature.WEBSITE.value in entity_features:
        return "ORG"
    return "MISC"


def generate_entities(
    raw_entities: dict[str, dict[EntityFeature, str]], countries: dict
) -> dict[str, EntityData]:
    """Generate entity entities (or description) that can be used in Entity Linking."""
    entities: dict[str, EntityData] = {}

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
    )
    logger.info("Generating entities")
    for ent_id, ent_feat in tqdm(raw_entities.items()):
        if EntityFeature.NAME in ent_feat:
            name: str | None = ent_feat.get(EntityFeature.NAME)

            if not name:
                continue

            if filter_bearer(name.strip()):
                entity_type = get_entity_type(ent_feat)
                text = name

                if entity_type == "ORG":
                    if desc := ent_feat.get(EntityFeature.ADDRESS):
                        text += ", located at " + desc
                    if desc := ent_feat.get(EntityFeature.DUNS_NUMBER):
                        text += ", DUNS " + desc
                    if desc := ent_feat.get(EntityFeature.PHONE):
                        text += ", phone " + desc
                    if desc := ent_feat.get(EntityFeature.COUNTRY_OF_ASSOCIATION):
                        country: str | None = get_country(countries, desc)
                        if country:
                            text += ", in " + country
                    if desc := ent_feat.get(EntityFeature.WEBSITE):
                        text += ", website " + desc
                    entities[str(ent_id)] = EntityData(
                        entity_id=str(ent_id), type=entity_type, name=name, description=text
                    )

                elif entity_type == "PER":
                    if desc := ent_feat.get(EntityFeature.DOB):
                        text += ", born " + desc
                    if desc := ent_feat.get(EntityFeature.PHONE):
                        text += ", phone " + desc
                    if desc := ent_feat.get(EntityFeature.ADDRESS):
                        text += ", located at " + desc
                    if desc := ent_feat.get(EntityFeature.GROUP_ASSOCIATION):
                        text += ", associated with " + desc
                    if desc := ent_feat.get(EntityFeature.COUNTRY_OF_ASSOCIATION):
                        country: str | None = get_country(countries, desc)  # type:ignore[no-redef]
                        if country:
                            text += ", in " + country
                    entities[str(ent_id)] = EntityData(
                        entity_id=str(ent_id), type=entity_type, name=name, description=text
                    )

                else:
                    print(f"New entity type: {ent_feat.get(EntityFeature.RECORD_TYPE)}")

    return entities


def write_entities(
    summaries: dict[str, EntityData], filepath: str | pathlib.Path = "data/senzing/entities.jsonl"
):
    """Write the generated summaries to a file."""
    logger.info(f"Writing entities to: {filepath}")
    with open(filepath, "w") as outfile:
        for ent_id, summary in summaries.items():
            json.dump(summary, outfile)
            outfile.write("\n")


class AliasRawData(TypedDict):
    alias: str
    entity: int
    type: str


def load_aliases(
    icij_path: str | pathlib.Path = "data/ICIJ-entity-report-2024-06-21_12-04-57-std.json",
    include_possibly_related: bool = True,
) -> list[AliasRawData]:
    alias_records: list[AliasRawData] = []

    logger.info(f"Parsing Senzing results: {icij_path}")
    num_lines = sum(1 for _ in open(icij_path))

    with tqdm(total=num_lines) as pbar:
        with open(icij_path, "r", encoding="utf-8") as fp:
            while line := fp.readline():
                dat = json.loads(line.strip())
                entity: dict = dat["RESOLVED_ENTITY"]
                related_entities: dict = dat["RELATED_ENTITIES"]

                if not entity["ENTITY_NAME"]:
                    continue

                entity_type = get_entity_type(entity["FEATURES"])

                # add aliases from resolved entities
                for record in entity["RECORDS"]:
                    alias_records.append(
                        {
                            "alias": record["ENTITY_DESC"],
                            "entity": record["INTERNAL_ID"],
                            "type": entity_type,
                        }
                    )

                # add aliases from related entities
                if not include_possibly_related:
                    continue
                for record in related_entities:
                    # MATCH_LEVEL_CODE is either POSSIBLY_SAME or POSSIBLY_RELATED or RESOLVED or DISCLOSED
                    # we choose to add an alias record if POSSIBLY_SAME
                    if record["MATCH_LEVEL_CODE"] in ["POSSIBLY_SAME", "RESOLVED", "DISCLOSED"]:
                        alias_records.append(
                            {
                                "alias": entity["ENTITY_NAME"],
                                "entity": record["ENTITY_ID"],
                                "type": entity_type,
                            }
                        )
                    # and discard if POSSIBLY_RELATED
                    elif record["MATCH_LEVEL_CODE"] == "POSSIBLY_RELATED":
                        continue

                pbar.update(1)

    return alias_records


class EntityRulerPattern(TypedDict):
    label: str
    pattern: str
    id: str


def generate_patterns(raw_aliases: list[AliasRawData]) -> list[EntityRulerPattern]:
    return [
        {
            "label": alias["type"],
            "pattern": alias["alias"],
            "id": str(alias["entity"]),
        }
        for alias in raw_aliases
    ]


def generate_aliases(raw_aliases: list[AliasRawData]) -> pd.DataFrame:
    logger.info("Generating aliases")
    df = (
        pd.DataFrame.from_records(raw_aliases)
        .astype({"entity": str})
        .groupby("alias")
        .agg(counts=("entity", Counter))
        .assign(entities=lambda d: d.counts.apply(list))
        .assign(
            probabilities=lambda d: d.counts.apply(
                lambda x: [count / x.total() for k, count in x.items()]
            )
        )
        .drop(columns="counts")
        .reset_index()
    )
    return df


def write_aliases(
    aliases: pd.DataFrame, filepath: str | pathlib.Path = "data/senzing/aliases.jsonl"
):
    logger.info(f"Writing aliases to: {filepath}")
    aliases.to_json(filepath, orient="records", lines=True)


def filter_senzing(
    raw_entities: dict[str, EntityData],
    raw_aliases: list[AliasRawData],
    patterns: list[EntityRulerPattern],
) -> tuple[dict[str, EntityData], list[AliasRawData]]:
    nlp = spacy.load(SPACY_MODEL, exclude=["ner"])

    logger.info("Loading patterns in EntityRuler")
    ruler = nlp.add_pipe("entity_ruler")
    with nlp.select_pipes(enable="tagger"):
        ruler.add_patterns(patterns)  # type: ignore

    logger.info("Loading dataset of articles")
    doc_bin = DocBin().from_disk(path="data/dataset.spacy")
    docs = list(doc_bin.get_docs(nlp.vocab))

    logger.info("Filtering Senzing results for entities matched by EntityRuler.")
    matched_names = set(ent.text for doc in nlp.pipe(docs) for ent in doc.ents)
    matched_ids = set(p["id"] for p in patterns if p["pattern"] in matched_names)

    # TODO: add friends of matched_ids

    filtered_entities = {k: v for k, v in raw_entities.items() if str(k) in matched_ids}
    filtered_aliases = [alias for alias in raw_aliases if str(alias["entity"]) in matched_ids]
    return filtered_entities, filtered_aliases


def main():
    """Entrypoint to the Senzing data pipeline."""
    countries = load_countries()
    raw_entities = load_entities()
    raw_aliases = load_aliases()

    patterns = generate_patterns(raw_aliases)
    stg_entities, stg_aliases = filter_senzing(raw_entities, raw_aliases, patterns)

    entities = generate_entities(stg_entities, countries)
    write_entities(entities)

    aliases = generate_aliases(stg_aliases)
    write_aliases(aliases)
