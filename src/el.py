# ref: https://colab.research.google.com/github/IBM/zshot/blob/examples/Zshot%20Example%20-%20Wikification.ipynb#scrollTo=LpMikizf5gk7
import spacy
import zshot
from zshot import PipelineConfig, displacy
from zshot.linker import LinkerRegen
from zshot.linker.linker_regen.utils import load_dbpedia_trie, load_wikipedia_trie
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.utils.mappings import spans_to_dbpedia, spans_to_wikipedia

dbpedia_trie = load_dbpedia_trie()
wikipedia_trie = load_wikipedia_trie()


nlp_wikipedia = spacy.load("en_core_web_sm")
nlp_config = PipelineConfig(
    mentions_extractor=MentionsExtractorSpacy(), linker=LinkerRegen(trie=wikipedia_trie)
)
nlp_wikipedia.add_pipe("zshot", config=nlp_config, last=True)


doc = nlp_wikipedia(
    "CH2O2 is a chemical compound similar to Acetamide used in International Business "
    "Machines Corporation (IBM)."
)
displacy.render(doc, style="ent")
print(list(zip(doc.ents, spans_to_wikipedia(doc._.spans))))


nlp_dbpedia = spacy.load("en_core_web_sm")
nlp_config = PipelineConfig(
    mentions_extractor=MentionsExtractorSpacy(), linker=LinkerRegen(trie=dbpedia_trie)
)
nlp_dbpedia.add_pipe("zshot", config=nlp_config, last=True)


doc = nlp_dbpedia(
    "CH2O2 is a chemical compound similar to Acetamide used in International Business "
    "Machines Corporation (IBM)."
)
displacy.render(doc, style="ent")
print(list(zip(doc.ents, spans_to_dbpedia(doc._.spans))))
