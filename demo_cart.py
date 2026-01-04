"""
CART (Classification and Regression Trees) Implementation Demonstration

This script demonstrates the implementation of the CART algorithm as described in:
C.E. Brodley and P.E. Utgoff, "Multivariate Decision Trees," Machine Learning, vol. 19, pp. 45-77, 1995.

Key Features of CART:
1. Uses Gini impurity as the splitting criterion (instead of entropy used in C4.5)
2. Creates binary decision trees (always splits into two child nodes)
3. Supports pruning to avoid overfitting
4. Handles both discrete and continuous attributes

The implementation includes:
- CartParameter: Parameter class for CART (similar to C45Parameter)
- CartNode: Binary tree node using Gini impurity for splitting
- Cart: Main CART classifier class
"""

from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet
from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter


def main():
    print("=" * 70)
    print("CART (Classification and Regression Trees) Implementation")
    print("=" * 70)
    print()
    
    # Load the Iris dataset
    print("Loading Iris dataset...")
    attributeTypes = 4 * [AttributeType.CONTINUOUS]
    dataDefinition = DataDefinition(attributeTypes)
    iris = DataSet(dataDefinition, ",", "datasets/iris.data")
    print(f"Dataset loaded: {iris.getInstanceList().size()} instances")
    print()
    
    # Train CART model
    print("Training CART model...")
    cart = Cart()
    # Parameters: seed=1, prune=True, crossValidationRatio=0.2
    cartParameter = CartParameter(1, True, 0.2)
    cart.train(iris.getInstanceList(), cartParameter)
    print("Training completed!")
    print()
    
    # Test the model
    print("Testing the model...")
    performance = cart.test(iris.getInstanceList())
    error_rate = 100 * performance.getErrorRate()
    accuracy = 100 * (1 - performance.getErrorRate())
    
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    
    # Make a prediction on a single instance
    print("Making prediction on first instance...")
    first_instance = iris.getInstanceList().get(0)
    prediction = cart.predict(first_instance)
    actual = first_instance.getClassLabel()
    print(f"Predicted class: {prediction}")
    print(f"Actual class: {actual}")
    print(f"Match: {'✓' if prediction == actual else '✗'}")
    print()
    
    print("=" * 70)
    print("Key Differences between CART and C4.5:")
    print("=" * 70)
    print("1. Splitting Criterion:")
    print("   - CART: Gini impurity (measures probability of misclassification)")
    print("   - C4.5: Information gain / entropy (measures information content)")
    print()
    print("2. Tree Structure:")
    print("   - CART: Always binary trees (two children per node)")
    print("   - C4.5: Can have multiple children per node")
    print()
    print("3. Attribute Handling:")
    print("   - CART: Binary splits for all attribute types")
    print("   - C4.5: Multi-way splits for discrete attributes")
    print()
    print("4. Historical Context:")
    print("   - CART: Developed by Breiman et al. (1984)")
    print("   - C4.5: Developed by Quinlan (1993)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
