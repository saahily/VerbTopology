# Verb Topology

This repo is my effort to try to figure out how verbs organize themselves semantically, across wildly different languages. my first NLP research project in a while!

## What is this

The basic idea: can we take a bunch of verb instances from text corpora and cluster them into meaningful semantic classes *without* any labels (minus the UD ones)? and different languages (english, telugu, vedic sanskrit, classical sanskrit) end up with similar verb categories?

This matters because:
- linguists have argued for decades about whether verb classes are universal or language-specific (spoiler: probably both?)
- I'm low-key obsessed with argument structure and how different languages encode who-does-what-to-whom
- also I want to eventually use this for my [conlang project](https://github.com/saahily/bhasha-nirmana) but that's a whole other thing

## Languages

Working with [Universal Dependencies](https://universaldependencies.org/introduction.html) (UD) treebanks (huge thanks to those who manually annotated them <3 they're the real ones):
- **English** (EWT) - the baseline
- **Telugu** (MTG) - Dravidian, very different morphology, curious what patterns emerge
- **Vedic Sanskrit** - has 40,000 sentences from various segments of the Vedic corpus (i.e. Ṛgveda, Śaunaka recension of the Atharvaveda, Maitrāyaṇīsaṃhitā from Yajurveda, and the Aitareya and Śatapatha Brāhmaṇas)
- **Sanskrit** (UFAL) - classical period, much smaller corpus sadly, from the Pancatantra

## Current status

**done:** verb extraction, syntactic feature engineering (phases 1-3)

### What works
- conllu parsing (the `conllu` library is great btw)
- verb instance extraction with all their arguments
- feature engineering for syntactic frames

### What's next
- contextual embeddings via XLM-R (need gpu time...)
- clustering comparison (hierarchical, graph-based, etc)
- verbnet alignment for english
- cross-linguistic analysis

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

the scripts in `src/` can be run directly.

## Directory structure

```
data/raw/         - UD treebank files (conllu format)
src/              - python modules
outputs/          - extracted features, cluster assignments, etc
```

## Notes

- the telugu corpus is pretty small (~1400 sentences). might not get great clusters but worth trying
- vedic verb morphology is... a lot. voice distinctions are really interesting though
- need to handle sandhi somehow? or just trust the tokenization
- look into whether XLM-R actually covers vedic or if it just pretends to

## References

- Universal Dependencies: https://universaldependencies.org/
- [Yamada et al 2021](https://aclanthology.org/2021.findings-acl.381.pdf?utm_source=consensus) has good stuff on contextual embeddings for verb clustering
- [VerbNet](https://verbs.colorado.edu/verbnet/) for evaluation, plus looking at natural dhātu verb classes in traditional Sanskrit grammatical tradition ??

---

this is a work in progress!! 

