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


def load_countries(country_codes_path: str | pathlib.Path = "data/senzing/country.tsv") -> dict:
    """Map from a country code to a full name."""
    COUNTRIES: dict = {}

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


def generate_entities(
    raw_entities: dict[str, dict[EntityFeature, str]], countries: dict
) -> dict[str, EntityData]:
    """Generate entity entities (or description) that can be used in Entity Linking."""
    entities: dict[str, EntityData] = {}

    for ent_id, ent_feat in raw_entities.items():
        if EntityFeature.NAME in ent_feat:
            name: str | None = ent_feat.get(EntityFeature.NAME)

            if not name:
                continue

            if filter_bearer(name.strip()):
                kind: str | None = ent_feat.get(EntityFeature.RECORD_TYPE)
                text = name

                if not kind:
                    continue

                elif kind == "ORGANIZATION":
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
                    entities[ent_id] = EntityData(
                        entity_id=ent_id, type=kind, name=name, description=text
                    )

                elif kind == "PERSON":
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
                    entities[ent_id] = EntityData(
                        entity_id=ent_id, type=kind, name=name, description=text
                    )

                else:
                    print(f"New entity type: {kind}")

    return entities


def write_entities(
    summaries: dict[str, EntityData], filepath: str | pathlib.Path = "data/senzing/entities.jsonl"
):
    """Write the generated summaries to a file."""
    with open(filepath, "w") as outfile:
        for ent_id, summary in summaries.items():
            json.dump(summary, outfile)
            outfile.write("\n")


class AliasRawData(TypedDict):
    alias: str
    entity: int


def load_aliases(
    icij_path: str | pathlib.Path = "data/ICIJ-entity-report-2024-06-21_12-04-57-std.json",
) -> list[AliasRawData]:
    alias_records: list[AliasRawData] = []

    with open(icij_path, "r", encoding="utf-8") as fp:
        while line := fp.readline():
            dat = json.loads(line.strip())
            ent: dict = dat["RESOLVED_ENTITY"]
            for record in ent["RECORDS"]:
                if ent["ENTITY_NAME"]:
                    alias_records.append(
                        {"alias": ent["ENTITY_NAME"], "entity": record["INTERNAL_ID"]}
                    )

    return alias_records


def generate_aliases(raw_aliases: list[AliasRawData]) -> pd.DataFrame:
    df = (
        pd.DataFrame.from_records(raw_aliases)
        .groupby("alias")
        .agg(counts=("entity", Counter))
        .assign(entities=lambda d: d.counts.apply(list))
        .assign(
            probabilites=lambda d: d.counts.apply(
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
    aliases.to_json(filepath, orient="records", lines=True)


def main():
    """Entrypoint to the Senzing data pipeline."""
    countries = load_countries()
    raw_entities = load_entities()
    entities = generate_entities(raw_entities, countries)
    write_entities(entities)

    raw_aliases = load_aliases()
    aliases = generate_aliases(raw_aliases)
    write_aliases(aliases)
