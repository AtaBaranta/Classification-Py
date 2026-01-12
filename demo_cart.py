"""
CART (Classification and Regression Trees) Implementation Demonstration
Supports both univariate and multivariate splits.
"""

from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet
from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun


def main():
    print("=" * 80)
    print("CART (Classification and Regression Trees) Demo")
    print("=" * 80)
    print()
    
    # Load Iris dataset
    attributeTypes = 4 * [AttributeType.CONTINUOUS]
    dataDefinition = DataDefinition(attributeTypes)
    iris = DataSet(dataDefinition, ",", "datasets/iris.data")
    print(f"Dataset: {iris.getInstanceList().size()} instances, 4 attributes")
    print()
    
    # Test 1: Univariate CART
    print("=" * 80)
    print("Univariate CART (single attribute splits)")
    print("=" * 80)
    cart_uni = Cart()
    cartParameter_uni = CartParameter(seed=1, prune=True, crossValidationRatio=0.2, multivariate=False)
    kfold = KFoldRun(10)
    experiment_uni = Experiment(cart_uni, cartParameter_uni, iris)
    performance_uni = kfold.execute(experiment_uni)
    
    error_rate_uni = 100 * performance_uni.meanPerformance().getErrorRate()
    accuracy_uni = 100 - error_rate_uni
    
    print(f"10-Fold CV Results:")
    print(f"  Error Rate: {error_rate_uni:.2f}%")
    print(f"  Accuracy: {accuracy_uni:.2f}%")
    print()
    
    # Test 2: Multivariate CART
    print("=" * 80)
    print("Multivariate CART (linear combination splits)")
    print("=" * 80)
    cart_multi = Cart()
    cartParameter_multi = CartParameter(seed=1, prune=True, crossValidationRatio=0.2, multivariate=True)
    experiment_multi = Experiment(cart_multi, cartParameter_multi, iris)
    performance_multi = kfold.execute(experiment_multi)
    
    error_rate_multi = 100 * performance_multi.meanPerformance().getErrorRate()
    accuracy_multi = 100 - error_rate_multi
    
    print(f"10-Fold CV Results:")
    print(f"  Error Rate: {error_rate_multi:.2f}%")
    print(f"  Accuracy: {accuracy_multi:.2f}%")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
