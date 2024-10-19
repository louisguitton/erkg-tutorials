import pandas as pd
import pytextrank  # noqa
import spacy
import srsly
from dagster import AssetSpec, Config, asset
from spacy.language import Language
from spacy.tokens import DocBin
from spacy_lancedb_linker.kb import AnnKnowledgeBase
from spacy_lancedb_linker.linker import AnnLinker  # noqa
from spacy_lancedb_linker.types import Alias, Entity

from src.analysis import analyse_el_docs
from src.scraper import SPACY_MODEL
from src.scraper import main as scraper_entrypoint
from src.senzing_pipeline import Entity as GraphEntity
from src.senzing_pipeline import (
    extract_senzing_results,
    filter_senzing,
    generate_aliases,
    generate_entities,
    load_aliases,
    load_countries,
    load_entities,
    write_aliases,
    write_entities,
)


class ICIJSenzingConfig(Config):
    senzing_results_path: str = "data/ICIJ-entity-report-2024-06-21_12-04-57-std.json"
    suspicions_path: str = "data/icij-example/suspicious.txt"
    country_codes_path: str = "data/senzing/country.tsv"
    spacy_dataset_path: str = "data/dataset.spacy"
    output_entities_jsonl_path: str = "data/icij-example/entities.jsonl"
    output_aliases_jsonl_path: str = "data/icij-example/aliases.jsonl"
    lancedb_uri: str = "data/sample-lancedb"


icij_senzing_results = AssetSpec(key="icij_senzing_results", group_name="source_dataset")


@asset(group_name="senzing_pipeline")
def suspicions(config: ICIJSenzingConfig) -> list[str]:
    with open(config.suspicions_path) as file:
        names = [line.rstrip() for line in file]
        return names


@asset(group_name="senzing_pipeline", deps=[icij_senzing_results])
def graph(config: ICIJSenzingConfig) -> dict[int, GraphEntity]:
    graph = extract_senzing_results(config.senzing_results_path)
    return graph


@asset(group_name="senzing_pipeline")
def suspicious_ids(suspicions: list[str], graph: dict[int, GraphEntity]) -> set[str]:
    return filter_senzing(suspicions, graph)


@asset(group_name="senzing_pipeline", deps=[icij_senzing_results])
def raw_entities(config: ICIJSenzingConfig):
    return load_entities(config.senzing_results_path)


@asset(group_name="senzing_pipeline", deps=[icij_senzing_results])
def raw_aliases(config: ICIJSenzingConfig):
    return load_aliases(config.senzing_results_path)


@asset(group_name="senzing_pipeline")
def filtered_entities(suspicious_ids, raw_entities):
    return {k: v for k, v in raw_entities.items() if str(k) in suspicious_ids}


@asset(group_name="senzing_pipeline")
def filtered_aliases(suspicious_ids: set[str], raw_aliases):
    return [alias for alias in raw_aliases if str(alias["entity"]) in suspicious_ids]


@asset(group_name="senzing_pipeline")
def countries(config: ICIJSenzingConfig) -> dict:
    return load_countries(config.country_codes_path)


@asset(group_name="entity_linking_inputs")
def entities_jsonl(
    config: ICIJSenzingConfig,
    filtered_entities,
    countries: dict,
) -> None:
    entities = generate_entities(filtered_entities, countries)
    write_entities(entities, config.output_entities_jsonl_path)


@asset(group_name="entity_linking_inputs")
def aliases_jsonl(config: ICIJSenzingConfig, filtered_aliases) -> None:
    aliases = generate_aliases(filtered_aliases)
    write_aliases(aliases, config.output_aliases_jsonl_path)


@asset(group_name="entity_linking_inputs")
def spacy_dataset(config: ICIJSenzingConfig) -> DocBin:
    return scraper_entrypoint(config.spacy_dataset_path)


@asset(group_name="spacy_pipeline")
def nlp() -> spacy.Language:
    return spacy.load(SPACY_MODEL)


@asset(
    group_name="spacy_pipeline",
    deps=[aliases_jsonl, entities_jsonl],
    io_manager_key="mem_io_manager",
)
def entity_linking(
    config: ICIJSenzingConfig, nlp: Language, spacy_dataset: DocBin
) -> list[pd.DataFrame]:
    entities = [Entity(**entity) for entity in srsly.read_jsonl(config.output_entities_jsonl_path)]

    aliases = [Alias(**alias) for alias in srsly.read_jsonl(config.output_aliases_jsonl_path)] + [
        Alias(alias=entity.name, entities=[entity.entity_id], probabilities=[1])
        for entity in entities
    ]

    ann_kb = AnnKnowledgeBase(uri=config.lancedb_uri)
    ann_kb.add_entities(entities)
    ann_kb.add_aliases(aliases)

    ann_linker = nlp.add_pipe("ann_linker", last=True)
    ann_linker.set_kb(ann_kb)  # type: ignore

    nlp.add_pipe("textrank")

    docs = spacy_dataset.get_docs(nlp.vocab)

    return analyse_el_docs(nlp.pipe(docs))
