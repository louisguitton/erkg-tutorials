"""
Parse Senzing ER results into a Neo4j Database.

Based on https://github.com/DerwenAI/ERKG/blob/main/examples/graph.ipynb
"""

import json
import os
import pathlib
import typing
import zipfile
from dataclasses import dataclass, field

import dotenv
import pandas as pd
from graphdatascience import GraphDataScience
from tqdm import tqdm


@dataclass(order=False, frozen=False)
class Entity:
    """
    A data class representing a resolved entity.
    """

    entity_uid: id
    name: str
    num_recs: int
    records: typing.Dict[str, str] = field(default_factory=lambda: {})
    related: typing.Dict[int, dict] = field(default_factory=lambda: {})
    has_ref: bool = False
    has_ref: bool = False


def extract_senzing_results(export_path: pathlib.Path) -> dict[int, Entity]:
    """Parse the Senzing results."""
    entities: dict[int, Entity] = {}

    with zipfile.ZipFile(export_path, "r") as z:
        for filename in z.namelist():
            # the zip file contains __MACOSX artefacts
            if filename.startswith("__MACOSX"):
                continue

            with z.open(filename) as fp:
                for line in tqdm(fp.readlines(), desc="read JSON"):
                    entity_dat: dict = json.loads(line)
                    entity_uid: int = entity_dat["RESOLVED_ENTITY"]["ENTITY_ID"]

                    entity_name: str = ""
                    records: dict[str, str] = {}

                    for rec in entity_dat["RESOLVED_ENTITY"]["RECORDS"]:
                        record_uid: str = ".".join(
                            [rec["DATA_SOURCE"].upper(), str(rec["RECORD_ID"])]
                        )
                        match_key: str = rec["MATCH_KEY"]

                        if match_key.strip() == "":
                            match_key = "INITIAL"
                        records[record_uid] = match_key

                        if entity_name == "" and rec["ENTITY_DESC"] != "":
                            entity_name = rec["ENTITY_DESC"]

                    if entity_name == "":
                        entity_name = entity_uid

                    entities[entity_uid] = Entity(
                        entity_uid=entity_uid,
                        name=entity_name,
                        records=records,
                        num_recs=len(records),
                        related={r["ENTITY_ID"]: r for r in entity_dat["RELATED_ENTITIES"]},
                    )

    for entity in entities.values():
        if entity.num_recs > 0:
            entity.has_ref = True

        for rel_ent_id in entity.related:
            entities[rel_ent_id].has_ref = True

    return entities


def get_neo4j(
    bolt_uri: str = None, database: str = None, username: str = None, password: str = None
) -> GraphDataScience:
    dotenv.load_dotenv(dotenv.find_dotenv())

    bolt_uri: str = bolt_uri or os.environ.get("NEO4J_BOLT")
    username: str = username or os.environ.get("NEO4J_USERNAME")
    password: str = password or os.environ.get("NEO4J_PASSWORD")
    database: str = database or os.environ.get("NEO4J_DATABASE")
    print(bolt_uri, username, password, database)

    gds: GraphDataScience = GraphDataScience(
        bolt_uri,
        auth=(
            username,
            password,
        ),
        database=database,
        aura_ds=False,
    )
    return gds


def load_to_neo4j(gds: GraphDataScience, entities: dict[int, Entity]) -> GraphDataScience:
    _setup_neo4j(gds)
    _populate_nodes_from_senzing_entities(gds, entities)
    _connect_resolved_records(gds, entities)
    _connect_related_entities(gds, entities)
    return gds


def _setup_neo4j(gds: GraphDataScience):
    gds.run_cypher(
        """
    DROP CONSTRAINT `entity_node_key` IF EXISTS
    """
    )

    gds.run_cypher(
        """
    CREATE CONSTRAINT `entity_node_key` IF NOT EXISTS
    FOR (ent:Entity)
    REQUIRE ent.uid IS NODE KEY
    """
    )


def _populate_nodes_from_senzing_entities(gds: GraphDataScience, entities: dict[int, Entity]):
    df_ent: pd.DataFrame = pd.DataFrame(
        [
            {
                "uid": entity.entity_uid,
                "name": entity.name,
                "has_ref": entity.has_ref,
            }
            for entity in entities.values()
        ]
    )

    unwind_query: str = """
    UNWIND $rows AS row
    CALL {
    WITH row
    MERGE (ent:Entity {uid: row.uid, name: row.name, has_ref: row.has_ref})
    } IN TRANSACTIONS OF 10000 ROWS
        """

    gds.run_cypher(
        unwind_query,
        {"rows": df_ent.to_dict(orient="records")},
    )


def _connect_resolved_records(gds: GraphDataScience, entities: dict[int, Entity]):
    df_rec: pd.DataFrame = pd.DataFrame(
        [
            {
                "entity_uid": entity.entity_uid,
                "record_uid": record_uid,
                "match_key": match_key,
            }
            for entity in entities.values()
            for record_uid, match_key in entity.records.items()
        ]
    )

    unwind_query: str = """
    UNWIND $rows AS row
    CALL {
    WITH row
    MATCH
        (ent:Entity {uid: row.entity_uid}),
        (rec:Record {uid: row.record_uid})
    MERGE (ent)-[:RESOLVES {match_key: row.match_key}]->(rec)
    } IN TRANSACTIONS OF 10000 ROWS
        """

    gds.run_cypher(
        unwind_query,
        {"rows": df_rec.to_dict(orient="records")},
    )


def _connect_related_entities(gds: GraphDataScience, entities: dict[int, Entity]):
    df_rel: pd.DataFrame = pd.DataFrame(
        [
            {
                "entity_uid": entity.entity_uid,
                "rel_ent": rel_ent["ENTITY_ID"],
                "ambiguous": (rel_ent["IS_AMBIGUOUS"] == 0),
                "disclosed": (rel_ent["IS_DISCLOSED"] == 0),
                "match_level": rel_ent["MATCH_LEVEL"],
                "match_level_code": rel_ent["MATCH_LEVEL_CODE"],
            }
            for entity in entities.values()
            for rel_key, rel_ent in entity.related.items()
        ]
    )

    unwind_query: str = """
UNWIND $rows AS row
CALL {
  WITH row
  MATCH
    (ent:Entity {uid: row.entity_uid}),
    (rel_ent:Entity {uid: row.rel_ent})
  MERGE (ent)-[:RELATED {ambiguous: row.ambiguous, disclosed: row.disclosed, match_level: row.match_level, match_level_code: row.match_level_code}]->(rel_ent)
} IN TRANSACTIONS OF 10000 ROWS
    """

    gds.run_cypher(
        unwind_query,
        {"rows": df_rel.to_dict(orient="records")},
    )


if __name__ == "__main__":
    # from src.senzing_parser import extract_senzing_results, get_neo4j, load_to_neo4j

    gds = get_neo4j()

    entities = extract_senzing_results("data/ICIJ-entity-report-2024-06-21_12-04-57-std.json.zip")

    load_to_neo4j(gds, entities)

    gds.run_cypher(
        """
MATCH (ent:Entity)
RETURN COUNT(ent.uid)
"""
    )
