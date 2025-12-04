# English Verb Clustering: Intrinsic Evaluation

Sorted by Silhouette score (higher is better).

| Method | Feature | k | Silhouette | CH Index | DB Index | Modularity |
|--------|---------|---|------------|----------|----------|------------|
| hdbscan | syntactic | 14 | 0.533 | 106.0 | 0.621 | - |
| hdbscan | joint | 4 | 0.467 | 131.8 | 1.025 | - |
| hdbscan | syntactic | 5 | 0.463 | 131.6 | 0.970 | - |
| hdbscan | joint | 2 | 0.268 | 12.6 | 1.076 | - |
| hierarchical_average | joint | 10 | 0.198 | 84.9 | 1.297 | - |
| kmeans | syntactic | 10 | 0.171 | 151.1 | 1.574 | - |
| hierarchical_average | syntactic | 10 | 0.171 | 90.5 | 1.541 | - |
| kmeans | joint | 10 | 0.162 | 139.6 | 1.667 | - |
| hierarchical_ward | joint | 10 | 0.156 | 120.6 | 1.673 | - |
| hierarchical_average | embedding | 10 | 0.153 | 2.5 | 0.875 | - |
| kmeans | syntactic | 50 | 0.151 | 62.1 | 1.631 | - |
| kmeans | syntactic | 20 | 0.144 | 102.9 | 1.644 | - |
| hierarchical_average | syntactic | 20 | 0.144 | 63.3 | 1.360 | - |
| kmeans | syntactic | 30 | 0.141 | 82.4 | 1.688 | - |
| hierarchical_average | joint | 30 | 0.134 | 51.8 | 1.367 | - |
| hierarchical_average | joint | 20 | 0.131 | 58.2 | 1.274 | - |
| kmeans | joint | 30 | 0.131 | 76.4 | 1.763 | - |
| hierarchical_ward | syntactic | 10 | 0.130 | 127.1 | 1.735 | - |
| kmeans | joint | 20 | 0.130 | 95.5 | 1.756 | - |
| hierarchical_average | syntactic | 30 | 0.129 | 48.9 | 1.277 | - |
| hierarchical_ward | syntactic | 50 | 0.126 | 58.3 | 1.604 | - |
| kmeans | joint | 50 | 0.124 | 56.4 | 1.642 | - |
| louvain | joint | 18 | 0.122 | 94.1 | 1.720 | 0.617 |
| hierarchical_average | syntactic | 50 | 0.119 | 38.7 | 1.229 | - |
| louvain | joint | 8 | 0.116 | 120.4 | 1.944 | 0.641 |
| hierarchical_average | joint | 50 | 0.115 | 38.0 | 1.292 | - |
| chinese_whispers | joint | 16 | 0.113 | 92.4 | 1.594 | - |
| hierarchical_ward | syntactic | 30 | 0.111 | 74.5 | 1.763 | - |
| hierarchical_average | embedding | 20 | 0.108 | 2.5 | 0.990 | - |
| hierarchical_ward | syntactic | 20 | 0.107 | 89.5 | 1.748 | - |