# Random Forests

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode (classification) or mean (regression) of the individual trees' predictions.

## Core Concept

The algorithm creates diversity among trees through two key mechanisms:

### Bootstrap Aggregating (Bagging)
Each tree is trained on a random sample of the training data, drawn with replacement. This means:
- Each tree sees a slightly different dataset
- On average, each tree uses about 63% of the training examples
- The remaining 37% form the "out-of-bag" sample used for validation

### Feature Randomness
At each split in a tree, only a random subset of features is considered. Typically:
- For classification: √n features (where n is total features)
- For regression: n/3 features
- This decorrelates the trees and reduces overfitting

## Algorithm Steps

1. Select a random sample (with replacement) from the training data
2. Build a decision tree on this sample
3. At each node, randomly select a subset of features
4. Choose the best split from these features
5. Grow the tree to maximum depth (no pruning)
6. Repeat steps 1-5 for N trees
7. Aggregate predictions by voting (classification) or averaging (regression)

## Advantages

- **Robust to Overfitting**: Averaging reduces variance without increasing bias
- **Handles Missing Data**: Can maintain accuracy with missing values
- **No Feature Scaling**: Works directly with raw features
- **Parallel Training**: Each tree can be built independently
- **Feature Importance**: Provides interpretable importance scores

## Disadvantages

- **Model Size**: Large forests require significant memory
- **Prediction Speed**: Slower than single trees for inference
- **Less Interpretable**: Harder to understand than a single decision tree
- **Biased with Imbalanced Data**: May favor majority classes

## Hyperparameters

Key parameters to tune:
- **n_estimators**: Number of trees (typically 100-500)
- **max_depth**: Maximum tree depth (controls overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **max_features**: Number of features to consider at each split
- **bootstrap**: Whether to use bootstrap samples

## Use Cases

Random Forests work particularly well for:
- Tabular data with mixed feature types
- Problems where interpretability is moderately important
- Scenarios requiring robust out-of-the-box performance
- Feature selection and importance ranking
