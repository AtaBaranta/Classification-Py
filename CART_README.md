# CART (Classification and Regression Trees) Implementation

## Overview

This implementation adds the CART (Classification and Regression Trees) algorithm to the Classification-Py library. CART is a decision tree algorithm described in:

> C.E. Brodley and P.E. Utgoff, "Multivariate Decision Trees," Machine Learning, vol. 19, pp. 45-77, 1995.

## Files Added

### 1. `Classification/Parameter/CartParameter.py`
Parameter class for the CART algorithm, containing:
- `seed`: Random seed for reproducibility
- `prune`: Boolean flag for pruning
- `crossValidationRatio`: Ratio for train/prune data split

### 2. `Classification/Model/DecisionTree/CartNode.py`
Binary decision tree node implementation with:
- **Gini Impurity**: Uses Gini index as splitting criterion instead of entropy
  - Gini(D) = 1 - Σ(p_i²) where p_i is the probability of class i
- **Binary Splits**: Always creates exactly 2 child nodes
  - For continuous attributes: threshold-based splits (value < threshold vs. value ≥ threshold)
  - For discrete attributes: binary splits (value = X vs. value ≠ X)
  - For indexed attributes: index-based binary splits

### 3. `Classification/Model/DecisionTree/Cart.py`
Main CART classifier that extends `ValidatedModel`:
- Training with optional pruning (20% validation, 80% training)
- Prediction for new instances
- Probability distribution estimation
- Model persistence (load/save)

### 4. `test/Classifier/CartTest.py`
Unit tests verifying the implementation on multiple datasets:
- Iris (continuous attributes)
- Bupa (continuous attributes)
- Dermatology (continuous attributes)
- Car (discrete attributes)
- Tic-tac-toe (discrete attributes)

### 5. `demo_cart.py`
Demonstration script showing:
- How to use the CART implementation
- Comparison with C4.5
- Performance metrics

## Key Differences: CART vs. C4.5

| Aspect | CART | C4.5 |
|--------|------|------|
| **Splitting Criterion** | Gini impurity | Information gain (entropy) |
| **Tree Structure** | Binary (2 children) | Multi-way (n children) |
| **Discrete Attributes** | Binary splits | One child per value |
| **Continuous Attributes** | Threshold splits | Threshold splits |
| **Pruning** | Cost-complexity | Error-based |
| **Developer** | Breiman et al. (1984) | Quinlan (1993) |

## Mathematical Background

### Gini Impurity

For a dataset D with K classes, the Gini impurity is:

```
Gini(D) = 1 - Σ(p_i²)
```

where p_i is the proportion of instances in class i.

### Splitting Criterion

For a binary split that divides D into D_left and D_right:

```
Gini_split = (|D_left|/|D|) × Gini(D_left) + (|D_right|/|D|) × Gini(D_right)
```

The algorithm chooses the split that minimizes Gini_split.

## Usage Example

```python
from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet
from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter

# Load dataset
attributeTypes = 4 * [AttributeType.CONTINUOUS]
dataDefinition = DataDefinition(attributeTypes)
dataset = DataSet(dataDefinition, ",", "datasets/iris.data")

# Train CART model
cart = Cart()
cartParameter = CartParameter(seed=1, prune=True, crossValidationRatio=0.2)
cart.train(dataset.getInstanceList(), cartParameter)

# Make predictions
instance = dataset.getInstanceList().get(0)
prediction = cart.predict(instance)
probability_dist = cart.predictProbability(instance)

# Evaluate performance
performance = cart.test(dataset.getInstanceList())
accuracy = 100 * (1 - performance.getErrorRate())
print(f"Accuracy: {accuracy:.2f}%")
```

## Running Tests

From the project root directory:

```bash
cd test/Classifier
PYTHONPATH=../.. python3 CartTest.py
```

Or run the demonstration:

```bash
python3 demo_cart.py
```

## Implementation Details

### Minimal Changes Approach

The implementation follows the existing codebase patterns:
- Uses the same class hierarchy (extends `ValidatedModel`)
- Follows the same parameter structure (similar to `C45Parameter`)
- Reuses existing utilities (`InstanceList`, `Partition`, `DecisionCondition`)
- Maintains compatibility with existing test infrastructure

### Binary Split Strategy

For discrete attributes with multiple values, CART creates binary splits:
- One child for instances where attribute = specific_value
- Another child for instances where attribute ≠ specific_value
- The algorithm tries all possible values and picks the best split

### Pruning

Similar to C4.5, CART supports pruning:
- Splits data into training (80%) and validation (20%)
- Builds full tree on training data
- Prunes nodes if validation accuracy improves
- Uses recursive post-pruning

## Performance

Test results on standard datasets (with pruning enabled):

| Dataset | Instances | Attributes | Error Rate |
|---------|-----------|------------|------------|
| Iris | 150 | 4 continuous | < 10% |
| Bupa | 345 | 6 continuous | < 45% |
| Dermatology | 366 | 34 continuous | < 10% |
| Car | 1,728 | 6 discrete | < 35% |
| Tic-tac-toe | 958 | 9 discrete | < 35% |

## References

1. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole Advanced Books & Software.

2. Brodley, C. E., & Utgoff, P. E. (1995). Multivariate Decision Trees. *Machine Learning*, 19, 45-77.

3. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers.

## License

This implementation follows the same license as the Classification-Py library.
