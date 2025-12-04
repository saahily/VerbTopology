# Verb Topology

This repo is my effort to try to figure out how verbs organize themselves semantically, across wildly different languages. my first NLP research project in a while!

## What is this

The basic idea: can we take a bunch of verb instances from text corpora and cluster them into meaningful semantic classes *without* any labels (minus the UD ones)? and would different languages (english, telugu, vedic sanskrit, classical sanskrit) end up with similar verb clusters?

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

## Pipeline
1. **parse treebanks** → grab every verb and what arguments it takes (subject? object? oblique with "to"?)
2. **build features** → each verb lemma becomes a vector based on its syntactic behavior across all instances
3. **get embeddings** → also extract XLM-R contextual embeddings for the semantic side
4. **cluster** → use various clustering algorithms (e.g. hierarchical, louvain, etc) and see what groups emerge
5. **evaluate** → compare english clusters against VerbNet classes
6. **compare cross-linguistically** → do similar clusters show up across english/telugu/vedic/sanskrit? (what I'm especially interested in)

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

