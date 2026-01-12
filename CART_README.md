# Multivariate CART (Classification and Regression Trees) Implementation

## Overview

This implementation adds **multivariate CART** (Classification and Regression Trees) algorithm to the Classification-Py library, as described in:

> Brodley, C. E., & Utgoff, P. E. (1995). "Multivariate Decision Trees," *Machine Learning*, vol. 19, pp. 45-77.

**Key Innovation:** Unlike standard decision trees that use axis-parallel splits (single attribute per split), multivariate CART uses **oblique splits** with linear combinations of attributes: `w₁x₁ + w₂x₂ + ... + wₙxₙ ≤ θ`

This allows for more flexible decision boundaries, particularly beneficial for datasets with correlated features.

## Files Added/Modified

### 1. `Classification/Attribute/LinearCombinationAttribute.py` ⭐ NEW
Represents linear combinations of attributes for multivariate splits:
- Stores weights for each attribute: `w₁x₁ + w₂x₂ + ... + wₙxₙ`
- Stores threshold value: `θ`
- Used for multivariate decision boundaries

### 2. `Claoblique decision boundaries

### 2. `Classification/Parameter/CartParameter.py`
Parameter class for the CART algorithm, containing:
- `seed`: Random seed for reproducibility
- `prune`: Boolean flag for pruning
- `crossValidationRatio`: Ratio for train/prune data split

### 3. `Classification/Model/DecisionTree/DecisionCondition.py`
Extended to support linear combination conditions:
- Handles traditional attribute comparisons
- Supports linear combinations: `Σ(wᵢxᵢ) ≤ θ` ⭐ NEW

### 4. `Classification/Model/DecisionTree/CartNode.py`
Multivariate binary decision tree node implementation:
- **Gini Impurity**: Uses Gini index as splitting criterion
  - Gini(D) = 1 - Σ(p_i²) where p_i is the probability of class i
- **Binary Splits**: Always creates exactly 2 child nodes
  
**Multivariate splits (primary):**
  - Linear combinations of continuous attributes using Linear Discriminant Analysis (LDA)
  - Split condition: `w₁x₁ + w₂x₂ + ... + wₙxₙ ≤ θ`
  - Learns optimal weights via LDA for binary class splits
  - Extends to multi-class via pairwise class comparisons

**Fallback splits:**
  - For discrete attributes: binary splits (value = X vs. value ≠ X)
  - For indexed attributes: index-based binary splits
  - For continuous attributes when LDA fails: threshold-based splits

### 5. `Classification/Model/DecisionTree/Cart.py`
Main CART classifier that extends `ValidatedModel`:
- Training with optional pruning (20% validation, 80% training)
- Uses multivariate splits by default
- Prediction for new instances
- Probability distribution estimation
- Model persistence (load/save)

### 6. `test/Classifier/CartTest.py`
Unit tests with **10-fold cross-validation** for realistic performance estimates:
- Tests multivariate CART on multiple datasets
- Prevents overfitting by using proper train/test splits
- Datasets: Iris, Bupa, Dermatology (continuous attributes)

### 7. `demo_cart.

### Multivariate CART vs. C4.5

| Aspect | Multivariate CART | C4.5 |
|--------|-------------------|------|
| **Split Condition** | Linear combination: `Σ(wᵢxᵢ) ≤ θ` | Single attribute: `xᵢ ≤ θ` |
| **Decision Boundary** | Oblique (any angle) | Axis-parallel |
| **Splitting Criterion** | Gini impurity | Information gain (entropy) |
| **Tree Structure** | Binary (2 children) | Multi-way (n children) |
| **Discrete Attributes** | Binary splits | One child per value |
| **Learning Method** | LDA for weights | Greedy search |
| **Best For** | Correlated features | Simple patterns |
| **Interpretability** | Lower | Higher
| **Learning Method** | Greedy search | LDA for weights |
| **Best For** | Simple patterns | Complex, correlated features |
| **Speed** | Faster | Slower |
| **Interpretability** | More interpretable | Less interpretable |

## Mathematical Background

### Gini Impurity

For a dataset D with K classes, the Gini impurity is:

```
Gini(D) = 1 - Σ(p_i²)
```

where p_i is the proportion of instances in class i.

### Univariate Splitting Criterion

For a binary split that divides D into D_left and D_right:

```
Gini_split = (|D_left|/|D|) × Gini(D_left) + (|D_right|/|D|) × Gini(D_right)
```Multivariate Splitting Criterion

For multivariate splits, we use Linear Discriminant Analysis (LDA) to find optimal weights.

**Multivariate split condition:**
```
w₁x₁ + w₂x₂ + ... + wₙxₙ ≤ θ
```

**LDA Weight Computation:**

For two classes C₁ and C₂ with means μ₁ and μ₂, and pooled covariance matrix Σ:

```
w = Σ⁻¹(μ₁ - μ₂)
θ = wᵀ · ((μ₁ + μ₂)/2)
```

For multi-class problems, we:
1. Try all pairs of classes (Cᵢ, Cⱼ)
2. Compute LDA weights for each pair
3. Evaluate split quality using Gini impurity on full dataset
4. Choose the split (multivariate or fallback) with lowest Gini

**Split Quality:**
```
Gini_split = (|D_left|/|D|) × Gini(D_left) + (|D_right|/|D|) × Gini(D_right)
```

where D_left contains instances with `Σ(wᵢxᵢ) ≤ θ` and D_right contains the rest.
from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet
from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter

# Load dataset
attributeTypes = 4 * [AttributeType.CONTINUOUS]
dataDefinition =

### Basic Usage

```python
from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet
from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter

# Load dataset with continuous attributes
attributeTypes = 4 * [AttributeType.CONTINUOUS]
dataDefinition = DataDefinition(attributeTypes)
dataset = DataSet(dataDefinition, ",", "datasets/iris.data")

# Train multivariate CART model
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

### Proper Evaluation with Cross-Validation

```python
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun

# Load dataset
attributeTypes = 4 * [AttributeType.CONTINUOUS]
dataDefinition = DataDefinition(attributeTypes)
dataset = DataSet(dataDefinition, ",", "datasets/iris.data")

# Train with 10-fold cross-validation
cart = Cart()
cartParameter = CartParameter(seed=1, prune=True, crossValidationRatio=0.2)

kfold = KFoldRun(10)
experiment = Experiment(cart, cartParameter, dataset
python3 -m unittest test.Classifier.CartTest

# Run only univariate CART tests
python3 -m unittest test.Classifier.CartTest.CartTest.test_Train

# Run only multivariate CART tests
python3 -m unittest test.Classifier.CartTest.CartTest.test_TrainMultivariate
```

Or run the demonstration:

```bash
python3 demo_cart.py
```

## Performance Expectations

### Important: Cross-Validation vs. Training Set Evaluation
specific test
python3 -m unittest test.Classifier.CartTest.CartTest.test_Train
```

Or run the demonstration:

```bash
python3 demo_cart.py
```

## Performance Expectations

### Important: Cross-Validation for Realistic Estimates

**❌ Wrong Approach (Overfitting):**
```python
# Train and test on same data - unrealistic!
cart.train(dataset.getInstanceList(), parameter)
performance = cart.test(dataset.getInstanceList())  # Too optimistic!
```

**✅ Correct Approach (Realistic):**
```python
# 10-fold cross-validation - proper evaluation
kfold = KFoldRun(10)
experiment = Experiment(cart, parameter, dataset)
performance = kfold.execute(experiment)  # Realistic estimate
```

### Expected Performance with 10-Fold Cross-Validation

| Dataset | Instances | Attributes | Expected Error Rate |
|---------|-----------|------------|-------------------|
| Iris | 150 | 4 continuous | ~5-7% |
| Bupa | 345 | 6 continuous | ~33-38% |
| Dermatology | 366 | 34 continuous | ~7-10% |

**Key Insight:** Multivariate CART performs well on datasets with correlated features because oblique decision boundaries can capture complex patterns that axis-parallel boundaries cannot
- Multiple attributes per split
- ObMultivariate Split Learning Algorithm

The core innovation is the use of LDA (Linear Discriminant Analysis) to learn oblique splits:

1. **For each pair of classes** (Cᵢ, Cⱼ):
   - Extract instances belonging to these classes
   - Compute class means: μ₁, μ₂
   - Compute pooled covariance matrix: Σ
   - Solve for LDA weights: w = Σ⁻¹(μ₁ - μ₂)
   - Compute threshold: θ = wᵀ · ((μ₁ + μ₂)/2)

2. **Evaluate split quality:**
   - Project all instances: z = Σ(wᵢxᵢ)
   - Split into left (z ≤ θ) and right (z > θ)
   - Compute Gini impurity

3. **Choose best split:**
   - Compare all pairwise multivariate splits
   - Compare with simple fallback splits
   - Select split with lowest Gini impurity

### Advantages of Oblique Splits

**Visual Comparison:**
```
Axis-Parallel (C4.5):          Oblique (Multivariate CART):
     
  │     ○ ○ ○                      │  ○ ○ ○
  │   ○ ○ ○ ○                      │ ○ ○ ○ ○
  │ ─────────                      │╱ ╱ ╱ ╱
  │   × × ×                       ╱│ × × ×
  │ × × × ×                     ╱  │× × × ×
  └──────────                   └──────────

  Requires multiple splits      Single oblique split
  to separate classes          captures the pattern
```

### Binary Split Strategy

- For continuous attributes: Uses LDA to find optimal linear combination
- For discrete attributes: Binary splits (value = X vs. value ≠ X)
- For indexed attributes: Index-based binary splits
- Always creates exactly 2 child nodes per split

### Pruning

Pruning prevents overfitting:
- Splits data into training (80%) and validation (20%)
- Builds full tree on training data
- Prunes nodes if validation accuracy improves
- Uses recursive post-pruning

### Numerical Stability

For robust multivariate splits:
- Adds regularization (1e-6 × I) to covariance matrix
- Handles singular matrices gracefully
- Falls back to simple splits if LDA fails
- Uses numpy for efficient
# 10-fold cros Analysis

### Comparison: Train-Test vs. Cross-Validation

This implementation emphasizes **proper evaluation methodology**:

**❌ Previous Approach (Overfitting):**
- Trained on entire dataset
- Tested on same dataset
- Result: Unrealistically high accuracy (>90% on everything)

**✅ Current Approach (Realistic):**
- 10-fold cross-validation
- Separate train/test splits
- Result: Realistic generalization performance

### Test Results with 10-Fold Cross-Validation

| Dataset | Instances | Attributes | Error Rate | Accuracy |
|---------|-----------|------------|------------|----------|
| Iris | 150 | 4 continuous | ~5-7% | ~93-95% |
| Bupa | 345 | 6 continuous | ~33-38% | ~62-67% |
| Dermatology | 366 | 34 continuous | ~7-10% | ~90-93% |

### Why Multivariate CART Works Well

**Advantages:**
- ✓ Captures complex, non-axis-aligned patterns
- ✓ Effective for datasets with correlated features
- ✓ Often requires fewer nodes than axis-parallel trees
- ✓ Can represent XOR-like patterns efficiently

**Trade-offs:**
- ✗ Less interpretable than simple trees
- ✗ Slower training (LDA computation)
- ✗ May overfit on small datasets without prun**Brodley, C. E., & Utgoff, P. E. (1995).** "Multivariate Decision Trees." *Machine Learning*, 19, 45-77.
   - Primary reference for multivariate CART
   - Describes oblique split learning via linear combinations

2. **Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).** *Classification and Regression Trees*. Wadsworth & Brooks/Cole.
   - Original CART algorithm (univariate)
   - Foundation for Gini impurity and pruning

3. **Quinlan, J. R. (1993).** *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers.
   - Comparison baseline (axis-parallel splits, information gain)