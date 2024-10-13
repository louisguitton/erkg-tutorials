from dagster import asset
from spacy.tokens import DocBin

from src.scraper import main as scraper_entrypoint
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


@asset(group_name="senzing_pipeline")
def senzing_results() -> str:
    return "data/ICIJ-entity-report-2024-06-21_12-04-57-std.json"


@asset(group_name="senzing_pipeline")
def suspicions() -> list[str]:
    with open("data/icij-example/suspicious.txt") as file:
        names = [line.rstrip() for line in file]
        return names


@asset(group_name="senzing_pipeline")
def graph(senzing_results):
    graph = extract_senzing_results(senzing_results)
    return graph


@asset(group_name="senzing_pipeline")
def suspicious_ids(suspicions, graph):
    return filter_senzing(suspicions, graph)


@asset(group_name="senzing_pipeline")
def raw_entities(senzing_results):
    return load_entities(senzing_results)


@asset(group_name="senzing_pipeline")
def raw_aliases(senzing_results):
    return load_aliases(senzing_results)


@asset(group_name="senzing_pipeline")
def filtered_entities(suspicious_ids, raw_entities):
    return {k: v for k, v in raw_entities.items() if str(k) in suspicious_ids}


@asset(group_name="senzing_pipeline")
def filtered_aliases(suspicious_ids, raw_aliases):
    return [alias for alias in raw_aliases if str(alias["entity"]) in suspicious_ids]


@asset(group_name="senzing_pipeline")
def countries():
    return load_countries()


@asset(group_name="entity_linking_inputs")
def entities_jsonl(filtered_entities, countries):
    entities = generate_entities(filtered_entities, countries)
    write_entities(entities)


@asset(group_name="entity_linking_inputs")
def aliases_jsonl(filtered_aliases):
    aliases = generate_aliases(filtered_aliases)
    write_aliases(aliases)


@asset(group_name="entity_linking_inputs")
def spacy_dataset() -> DocBin:
    return scraper_entrypoint()
