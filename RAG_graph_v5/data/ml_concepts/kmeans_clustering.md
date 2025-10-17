# K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions data into K distinct, non-overlapping clusters based on feature similarity. It's one of the simplest and most popular clustering algorithms.

## Algorithm

The K-Means algorithm follows an iterative refinement approach:

### Steps:
1. **Initialize**: Randomly select K data points as initial cluster centroids
2. **Assignment**: Assign each data point to the nearest centroid (using Euclidean distance)
3. **Update**: Recalculate centroids as the mean of all points assigned to each cluster
4. **Repeat**: Continue steps 2-3 until convergence (centroids stop moving)

### Convergence Criteria:
- Centroids change less than a threshold
- Maximum number of iterations reached
- No points change cluster assignment

## Distance Metrics

While Euclidean distance is standard, alternatives include:
- **Manhattan Distance**: Sum of absolute differences
- **Cosine Similarity**: Angle between vectors (for text data)
- **Mahalanobis Distance**: Accounts for feature correlations

## Choosing K

Determining the optimal number of clusters is crucial:

### Elbow Method:
Plot within-cluster sum of squares (WCSS) against K values. The "elbow" point suggests optimal K.

### Silhouette Score:
Measures how similar points are to their own cluster versus other clusters.
- Range: [-1, 1]
- Higher is better (well-separated clusters)

### Gap Statistic:
Compares WCSS to that of random data to identify natural clustering.

## Advantages

- **Simple and Fast**: Easy to implement, O(n·K·i) complexity
- **Scalable**: Works well with large datasets
- **Guaranteed Convergence**: Always converges to a local minimum
- **Interpretable**: Cluster centers are easily visualized and understood

## Disadvantages

- **Requires K a priori**: Must specify number of clusters beforehand
- **Sensitive to Initialization**: Different initial centroids yield different results
- **Spherical Clusters Assumption**: Struggles with irregular shapes
- **Outlier Sensitivity**: Outliers significantly affect centroid positions
- **Equal Cluster Size Bias**: Tends to create similar-sized clusters

## Improvements and Variants

### K-Means++:
Improved initialization strategy that spreads initial centroids apart, leading to better convergence.

### Mini-Batch K-Means:
Uses random subsets of data for updates, dramatically reducing computation time for large datasets.

### K-Medoids:
Uses actual data points as cluster centers, more robust to outliers.

## Hyperparameters

- **n_clusters (K)**: Number of clusters to form
- **init**: Initialization method ('random', 'k-means++')
- **max_iter**: Maximum number of iterations
- **n_init**: Number of times algorithm runs with different initializations
- **tol**: Convergence tolerance

## Applications

K-Means is widely used for:
- Customer segmentation in marketing
- Image compression (color quantization)
- Document clustering
- Anomaly detection
- Feature engineering (cluster membership as features)
- Gene sequence analysis
