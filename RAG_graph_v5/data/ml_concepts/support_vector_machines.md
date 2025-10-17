# Support Vector Machines (SVM)

Support Vector Machines are supervised learning models used for classification and regression tasks. They find the optimal hyperplane that maximizes the margin between different classes in the feature space.

## Core Principle

SVMs aim to find a decision boundary (hyperplane) that:
1. Separates classes with maximum margin
2. Is defined by support vectors (data points closest to the boundary)
3. Generalizes well to unseen data

### Mathematical Foundation

For linearly separable data, SVM solves:
- Maximize: margin = 2/||w||
- Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all training examples

Where:
- w is the weight vector perpendicular to the hyperplane
- b is the bias term
- xᵢ are training examples
- yᵢ are class labels (-1 or +1)

## The Kernel Trick

For non-linearly separable data, SVMs use kernel functions to implicitly map data to higher dimensions:

### Common Kernels:
- **Linear**: K(x, y) = x·y
- **Polynomial**: K(x, y) = (x·y + c)ᵈ
- **RBF (Radial Basis Function)**: K(x, y) = exp(-γ||x - y||²)
- **Sigmoid**: K(x, y) = tanh(αx·y + c)

The RBF kernel is most commonly used and can model complex decision boundaries.

## Soft Margin Classification

Real-world data is rarely perfectly separable. Soft margin SVMs introduce a penalty parameter C:
- **Large C**: Hard margin (low tolerance for misclassification)
- **Small C**: Soft margin (more tolerance, better generalization)

The optimization becomes:
- Minimize: ½||w||² + C·Σξᵢ
- Where ξᵢ are slack variables allowing misclassification

## Advantages

- **Effective in High Dimensions**: Works well when features >> samples
- **Memory Efficient**: Only stores support vectors
- **Versatile**: Different kernel functions for different data patterns
- **Robust**: Less prone to overfitting in high dimensions

## Disadvantages

- **Slow Training**: O(n²) to O(n³) time complexity for large datasets
- **No Probability Estimates**: Requires additional calibration (Platt scaling)
- **Sensitive to Scaling**: Features must be normalized
- **Kernel Selection**: Choosing and tuning the right kernel can be challenging

## Hyperparameters

Key parameters:
- **C**: Regularization parameter (controls margin hardness)
- **kernel**: Type of kernel function
- **gamma**: Kernel coefficient (for RBF, polynomial, sigmoid)
- **degree**: Polynomial degree (for polynomial kernel)

## Applications

SVMs excel at:
- Text classification and sentiment analysis
- Image recognition
- Bioinformatics (protein classification)
- Handwriting recognition
- Anomaly detection with one-class SVM
