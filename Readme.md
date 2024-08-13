# erkg-tutorials

> Tutorials for Entity Resolved Knowledge Graphs

## Installation

- setup python env
- download Senzing ER results
- setup local Neo4j with GDS plugin and [Macos setting](https://neo4j.com/docs/graph-data-science/current/installation/#_graph_data_science_on_macos)
- run src.senzing_parser

### "SessionExpired: Failed to read from defunct connection" errors

graphdatascience==1.10 (the Py library)
Neo4j Desktop 1.6.0
Neo4j DB engine  5.18.1
GDS plugin 2.6.8

### debug Neo4j running locally

```sh
lsof -i tcp:5000-5005,6000-6005,7000-7005
```
