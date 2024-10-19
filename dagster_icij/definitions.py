from dagster import Definitions, InMemoryIOManager, load_assets_from_modules

from dagster_icij import assets

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "mem_io_manager": InMemoryIOManager(),
    },
)
