# erkg-tutorials

> Tutorials for Entity Resolved Knowledge Graphs

## Installation

- setup python env
- download Senzing ER results
- setup local Neo4j with GDS plugin and [Macos setting](https://neo4j.com/docs/graph-data-science/current/installation/#_graph_data_science_on_macos)
- run src.senzing_parser

```python
brew doctor
# do missing methods from brew doctor
brew upgrade
# ref https://sebhastian.com/error-metadata-generation-failed/
pip install --upgrade pip setuptools wheel
pip install spacy-ann-linker
```

### "SessionExpired: Failed to read from defunct connection" errors

graphdatascience==1.10 (the Py library)
Neo4j Desktop 1.6.0
Neo4j DB engine  5.18.1
GDS plugin 2.6.8

### debug Neo4j running locally

```sh
lsof -i tcp:5000-5005,6000-6005,7000-7005
```

## TODO

- [x] read source code of zshot.linker.linker_regen.trie.Trie to understand format and if I can build my own
- [ ] make microsoft sacy ann linker work on example data
  - spacy_ann create_index en_core_web_md examples/tutorial/data examples/tutorial/models
- [ ] create entities.jsonl and aliases.jsonl using dbpedia entities linked from zshot and make microsoft work on custom data

- [ ] test Gliner mention extractor, test Gliner linker
- [ ] TODO: test relation extractor

## setup notes

```sh
pyenv install --list | grep " 3\.[(7)]"
brew install openssl readline sqlite3 xz zlib
pyenv install 3.7.17
pyenv local 3.7.17
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install flit
OPENBLAS="$(brew --prefix openblas)" MACOSX_DEPLOYMENT_TARGET=14.5 flit install --deps=all --symlink
spacy download en_core_web_md

brew install openblas
pip install --upgrade pip setuptools wheel
OPENBLAS="$(brew --prefix openblas)" MACOSX_DEPLOYMENT_TARGET=14.5 pip install spacy-ann-linker

export OPENBLAS=$(brew --prefix openblas)

export CFLAGS="-falign-functions=8 ${CFLAGS}"
```

Download and unzip the Senzing overlay:
<https://storage.googleapis.com/erkg/icij/ICIJ-entity-report-2024-06-21_12-04-57-std.json.zip>
