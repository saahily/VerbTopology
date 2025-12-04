# VerbTopology

trying to figure out how verbs organize themselves semantically, across wildly different languages. my first NLP research project in a while!

## what is this

the basic idea: can we take a bunch of verb instances from text corpora and cluster them into meaningful semantic classes *without* any labels? and different languages (english, telugu, vedic sanskrit, classical sanskrit) end up with similar verb categories?

this matters because:
- linguists have argued for decades about whether verb classes are universal or language-specific (spoiler: probably both?)
- i'm low-key obsessed with argument structure and how different languages encode who-does-what-to-whom
- also i want to eventually use this for my conlang project but that's a whole other thing

## languages

working with Universal Dependencies treebanks (huge thanks to those who manually annotated them <3 they're the real ones):
- **English** (EWT) - the baseline
- **Telugu** (MTG) - dravidian, very different morphology, curious what patterns emerge
- **Vedic Sanskrit** - has 40,000 sentences from various segments of the Vedic corpus (i.e. Ṛgveda, Śaunaka recension of the Atharvaveda, Maitrāyaṇīsaṃhitā from Yajurveda, and the Aitareya and Śatapatha Brāhmaṇas)
- **Sanskrit** (UFAL) - classical period, much smaller corpus sadly, from the Pancatantra

## current status

phases 1-3 done. extracted verbs from all treebanks, built syntactic feature vectors. next up is the actual clustering (finally).

### what works
- conllu parsing (the `conllu` library is great btw)
- verb instance extraction with all their arguments
- feature engineering for syntactic frames

### whats next
- contextual embeddings via XLM-R (need gpu time...)
- clustering comparison (hierarchical, graph-based, etc)
- verbnet alignment for english
- cross-linguistic analysis

## running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

the scripts in `src/` can be run directly. check IMPLEMENTATION_PLAN.md for details on the full pipeline.

## directory structure

```
data/raw/         - UD treebank files (conllu format)
src/              - python modules
outputs/          - extracted features, cluster assignments, etc
```

## notes

- the telugu corpus is pretty small (~1400 sentences). might not get great clusters but worth trying
- vedic verb morphology is... a lot. voice distinctions are really interesting though
- need to handle sandhi somehow? or just trust the tokenization
- look into whether XLM-R actually covers vedic or if it just pretends to

## references

- Universal Dependencies: https://universaldependencies.org/
- Yamada et al 2021 has good stuff on contextual embeddings for verb clustering
- VerbNet for evaluation

---

this is a work in progress!! 

