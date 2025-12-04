# English Verb Clustering: Intrinsic Evaluation

Sorted by Silhouette score (higher is better).

| Method | Feature | k | Silhouette | CH Index | DB Index | Modularity |
|--------|---------|---|------------|----------|----------|------------|
| hdbscan | syntactic | 14 | 0.533 | 106.0 | 0.621 | - |
| hdbscan | syntactic | 5 | 0.463 | 131.6 | 0.970 | - |
| kmeans | syntactic | 10 | 0.171 | 151.1 | 1.574 | - |
| hierarchical_average | syntactic | 10 | 0.171 | 90.5 | 1.541 | - |
| hierarchical_average | embedding | 10 | 0.153 | 2.5 | 0.875 | - |
| kmeans | syntactic | 50 | 0.151 | 62.1 | 1.631 | - |
| kmeans | syntactic | 20 | 0.144 | 102.9 | 1.644 | - |
| hierarchical_average | syntactic | 20 | 0.144 | 63.3 | 1.360 | - |
| kmeans | syntactic | 30 | 0.141 | 82.4 | 1.688 | - |
| hierarchical_average | joint | 10 | 0.138 | 2.8 | 1.150 | - |
| hierarchical_ward | syntactic | 10 | 0.130 | 127.1 | 1.735 | - |
| hierarchical_average | syntactic | 30 | 0.129 | 48.9 | 1.277 | - |
| hierarchical_ward | syntactic | 50 | 0.126 | 58.3 | 1.604 | - |
| louvain | syntactic | 18 | 0.124 | 96.1 | 1.696 | 0.621 |
| hierarchical_average | syntactic | 50 | 0.119 | 38.7 | 1.229 | - |
| hierarchical_ward | syntactic | 30 | 0.111 | 74.5 | 1.763 | - |
| hierarchical_average | embedding | 20 | 0.108 | 2.5 | 0.990 | - |
| hierarchical_ward | syntactic | 20 | 0.107 | 89.5 | 1.748 | - |
| spectral | syntactic | 20 | 0.100 | 89.5 | 1.596 | - |
| louvain | syntactic | 22 | 0.095 | 83.4 | 1.623 | 0.598 |
| hierarchical_average | joint | 20 | 0.094 | 2.6 | 1.149 | - |
| louvain | syntactic | 11 | 0.093 | 107.1 | 1.783 | 0.643 |
| louvain | syntactic | 15 | 0.080 | 87.4 | 1.792 | 0.617 |
| hierarchical_average | embedding | 30 | 0.078 | 2.4 | 0.994 | - |
| spectral | syntactic | 10 | 0.074 | 107.4 | 1.347 | - |
| hierarchical_average | joint | 30 | 0.068 | 2.4 | 1.005 | - |
| chinese_whispers | syntactic | 16 | 0.068 | 89.6 | 1.463 | - |
| hdbscan | joint | 2 | 0.047 | 1.8 | 2.701 | - |
| hierarchical_average | embedding | 50 | 0.040 | 2.3 | 1.047 | - |
| hierarchical_average | joint | 50 | 0.039 | 2.6 | 1.251 | - |