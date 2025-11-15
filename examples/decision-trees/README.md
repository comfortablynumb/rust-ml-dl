# Decision Trees Example

This example demonstrates classification using decision trees in Rust with the Linfa library.

## Overview

Decision trees are supervised learning models that make predictions by learning simple decision rules from data features. They create a tree-like model of decisions.

## How Decision Trees Work

A decision tree recursively splits the data based on feature values:

```
                [Feature1 < 5.0?]
                /              \
              Yes               No
              /                  \
        [Class A]          [Feature2 < 3.0?]
                           /              \
                         Yes               No
                         /                  \
                   [Class B]           [Class C]
```

Each path from root to leaf represents a decision rule.

## Splitting Criteria

### Gini Impurity

Measures the probability of incorrect classification:

```
Gini = 1 - Σᵢ pᵢ²
```

- `pᵢ` = proportion of samples belonging to class i
- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity (50/50 split in binary)

### Information Gain (Entropy)

Measures the reduction in entropy after a split:

```
Entropy = -Σᵢ pᵢ log₂(pᵢ)
Information Gain = Entropy(parent) - Weighted_Average(Entropy(children))
```

## Running the Example

```bash
cargo run --package decision-trees
```

## What the Example Does

1. **Generates synthetic multi-class data** with clear decision boundaries
2. **Trains decision trees** with different maximum depths (3, 5, 10)
3. **Compares performance** to show effect of depth on accuracy
4. **Evaluates final model** using confusion matrix and per-class metrics
5. **Shows sample predictions** with feature values

## Key Concepts

### Hyperparameters

- **max_depth**: Maximum depth of the tree
  - Too small: Underfitting (high bias)
  - Too large: Overfitting (high variance)

- **min_samples_split**: Minimum samples required to split a node
  - Larger values prevent overfitting

- **min_samples_leaf**: Minimum samples required at a leaf node
  - Controls granularity of decision boundaries

### Overfitting vs Underfitting

```
Depth too small (Underfitting):
- High training error
- High test error
- Model too simple

Optimal depth:
- Low training error
- Low test error
- Good generalization

Depth too large (Overfitting):
- Very low training error
- High test error
- Memorizes training data
```

### When to Use Decision Trees

- Need an interpretable model (can explain decisions)
- Mix of numerical and categorical features
- Non-linear decision boundaries
- Feature interactions are important
- Quick baseline model
- As building block for ensemble methods (Random Forest, XGBoost)

### Advantages

1. **Interpretable**: Easy to visualize and explain
2. **No preprocessing needed**: No feature scaling required
3. **Handles non-linearity**: Can model complex relationships
4. **Feature interactions**: Automatically captures interactions
5. **Handles mixed data**: Both numerical and categorical
6. **Fast predictions**: O(log n) prediction time

### Limitations

1. **Overfitting**: Can create overly complex trees
2. **Instability**: Small data changes can produce very different trees
3. **Greedy**: Makes locally optimal splits (not globally optimal)
4. **Biased splits**: Favors features with more levels
5. **Axis-aligned**: Struggles with diagonal decision boundaries

## Preventing Overfitting

1. **Pre-pruning** (during training):
   - Set `max_depth`
   - Set `min_samples_split`
   - Set `min_samples_leaf`
   - Set `max_leaf_nodes`

2. **Post-pruning** (after training):
   - Build full tree
   - Remove branches that don't improve validation performance

## Ensemble Methods

Single decision trees often overfit. Ensemble methods combine multiple trees:

### Random Forest
- Trains multiple trees on random subsets of data and features
- Averages predictions (regression) or votes (classification)
- Reduces variance while maintaining low bias

### Gradient Boosting (XGBoost, LightGBM)
- Trains trees sequentially
- Each tree corrects errors of previous trees
- Often achieves state-of-the-art performance

### AdaBoost
- Trains trees on weighted data
- Focuses on misclassified samples
- Combines weak learners into strong learner

## Visualization

Decision trees can be visualized to understand the learned rules:

```
if feature[0] < 5.0:
    if feature[1] < 3.0:
        predict Class A
    else:
        predict Class B
else:
    predict Class C
```

## Libraries Used

- `linfa`: Machine learning framework for Rust
- `linfa-trees`: Decision tree and random forest implementation
- `ndarray`: N-dimensional arrays for numerical computing

## Further Reading

- [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
- [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
- [Random Forests](https://en.wikipedia.org/wiki/Random_forest)
