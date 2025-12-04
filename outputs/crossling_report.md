# Cross-Linguistic Analysis Report

## Phase 7: Per-Language Clustering

| Language | Lemmas | Best k | Silhouette |
|----------|--------|--------|------------|
| **English** | 920 | 30 | 0.44 |
| **Telugu** | 97 | 5 | 0.31 |
| **Vedic Sanskrit** | 1421 | 5 | 0.20 |
| **Sanskrit (UFAL)** | 24 | 3 | 0.26 |

### Telugu Clusters (k=5)
- Small corpus (97 verbs), but clusters emerge
- Cluster 1: 32 verbs including ceppēḍu (say), ceyyāli (do), cūsina (see)
- Cluster 5: 25 verbs including lēnu/lēru (be not)

### Vedic Sanskrit Clusters (k=5)
- Large corpus (1421 verbs) with many compound verbs (prefix + root)
- Cluster 1: 538 verbs - mixed transitives
- Cluster 4: 66 verbs - includes speech verbs (abhivac, anubrū, anumantray)

### Sanskrit (UFAL) Clusters (k=3)
- Very small corpus (24 verbs)
- Cluster 2: 6 verbs including अस् (be), भू (become), स्था (stand) - stative verbs
- Cluster 3: 10 verbs including गम् (go), कृ (do), ग्रह् (take) - dynamic verbs

## Phase 8: Cross-Linguistic Comparison

### 1. Centroid Similarity
- English-Vedic cluster centroid similarity: **mean 0.99, max 0.99**
- Very high similarity suggests embeddings are in shared space

### 2. Joint Multilingual Clustering
- Pooled English + Vedic: 2341 verbs
- Joint clustering at k=50:
  - **Mixed clusters: 0** (no clusters contain both English and Vedic)
  - English-only: 28 clusters
  - Vedic-only: 22 clusters

### 3. Key Findings

**Finding 1: Language-Specific Clustering**
Despite high centroid similarity, joint clustering separates languages completely. This suggests:
- XLM-R embeddings have language-specific components
- Or: syntactic features (which differ by language) drive separation

**Finding 2: XLM-R Coverage Issues for Vedic**
Nearest-neighbor analysis shows poor semantic matches:
- English "go" matches Vedic "ākṛ" (to do) not "gam" (to go)
- This suggests XLM-R's Vedic Sanskrit coverage is limited

**Finding 3: Structural Similarities**
Despite clustering separately, both languages show similar patterns:
- Both have propositional attitude / cognition clusters
- Both have motion/dynamic vs. stative distinctions
- Both have speech/communication verb groups

## Implications for Research

1. **Cross-lingual verb classes exist** in the sense that similar semantic-syntactic patterns emerge independently in each language

2. **Direct embedding-space alignment is weak** for low-resource languages like Vedic Sanskrit

3. **Better approach**: Compare cluster *structures* and *patterns* rather than direct verb-to-verb alignment

## Files Generated
- `telugu_dendrogram_labeled.png`
- `vedic_dendrogram_labeled.png`  
- `sanskrit_dendrogram_labeled.png`
- `crossling_clustering_results.json`
- `crossling_analysis.json`

