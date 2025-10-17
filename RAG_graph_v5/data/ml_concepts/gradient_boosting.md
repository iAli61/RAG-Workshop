# Gradient Boosting

Gradient Boosting is an ensemble machine learning technique that builds a strong predictive model by combining multiple weak learners, typically decision trees, in a sequential manner.

## How It Works

The algorithm works by iteratively adding models that correct the errors made by previous models. Each new model is trained to predict the residual errors (gradients) of the ensemble so far.

### Key Steps:
1. Initialize the model with a constant value
2. For each iteration:
   - Calculate the residuals (errors) of the current ensemble
   - Train a new weak learner to predict these residuals
   - Add the new learner to the ensemble with a learning rate multiplier
3. Continue until a stopping criterion is met

## Advantages

- **High Accuracy**: Often achieves state-of-the-art performance on structured data
- **Flexibility**: Can optimize different loss functions
- **Feature Importance**: Provides insights into which features are most predictive
- **Handles Mixed Data**: Works well with both numerical and categorical features

## Disadvantages

- **Training Time**: Sequential nature makes it slower to train than parallel methods
- **Overfitting Risk**: Can overfit if not properly regularized
- **Hyperparameter Sensitivity**: Requires careful tuning of learning rate, tree depth, and number of estimators
- **Memory Usage**: Stores all trees in memory

## Common Implementations

- **XGBoost**: Extreme Gradient Boosting with optimizations
- **LightGBM**: Microsoft's fast gradient boosting framework
- **CatBoost**: Yandex's implementation with categorical feature support
