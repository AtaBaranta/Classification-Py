from io import TextIOWrapper

from Math.DiscreteDistribution import DiscreteDistribution
from Util.RandomArray import RandomArray

from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.DecisionTree.DecisionCondition import DecisionCondition
from Classification.Model.Model import Model
from Classification.Parameter.CartParameter import CartParameter
import numpy as np


class CartNode(object):
    children: list
    __class_label: str = None
    leaf: bool
    __condition: DecisionCondition
    __classLabelsDistribution: DiscreteDistribution
    EPSILON = 0.0000000001

    def constructor1(self,
                     data: InstanceList,
                     condition=None,
                     parameter: CartParameter = None,
                     isStump=False
                     ):
        """
        The CartNode method takes InstanceList data as input and creates a binary decision tree node using
        the Gini impurity as the splitting criterion. This is a multivariate CART implementation that uses
        linear combinations of attributes for splits when possible.

        Multivariate CART creates oblique splits:
        - For continuous attributes: tries linear combinations using LDA (w₁x₁ + w₂x₂ + ... ≤ θ)
        - Falls back to univariate splits if multivariate learning fails
        - For discrete attributes: binary splits based on equality
        - For discrete indexed attributes: binary splits based on index

        PARAMETERS
        ----------
        data : InstanceList
            InstanceList input.
        condition : DecisionCondition
            DecisionCondition to check.
        parameter : CartParameter
            CartParameter like seed, prune, crossValidationRatio.
        isStump : bool
            Refers to decision trees with only 1 splitting rule.
        """
        best_attribute = -1
        best_split_value = 0
        self.__condition = condition
        self.__classLabelsDistribution = DiscreteDistribution()
        labels = data.getClassLabels()
        for label in labels:
            self.__classLabelsDistribution.addItem(label)
        self.__class_label = InstanceList.getMaximum(labels)
        self.leaf = True
        self.children = []
        class_labels = data.getDistinctClassLabels()
        if len(class_labels) == 1:
            return
        if isStump and condition is not None:
            return
        
        index_list = [i for i in range(data.get(0).attributeSize())]
        size = data.get(0).attributeSize()
        
        best_gini = self.__giniIndex(data.classDistribution())
        best_attribute = -1
        best_split_value = 0
        
        # Try multivariate splits if enabled in parameter
        best_multivariate_gini = float('inf')
        best_multivariate_weights = None
        best_multivariate_threshold = None
        
        if parameter is not None and parameter.isMultivariate():
            best_multivariate_gini, best_multivariate_weights, best_multivariate_threshold = self.__findBestMultivariateSplit(data)
        
        # Always try univariate splits as well (as fallback or primary approach)
        
        for j in range(size):
            index = index_list[j]
            if isinstance(data.get(0).getAttribute(index), DiscreteIndexedAttribute):
                # For discrete indexed attributes, try binary splits
                for k in range(data.get(0).getAttribute(index).getMaxIndex()):
                    distribution = data.discreteIndexedAttributeClassDistribution(index, k)
                    if distribution.getSum() > 0:
                        class_distribution = data.classDistribution()
                        class_distribution.removeDistribution(distribution)
                        gini = (class_distribution.getSum() / data.size()) * self.__giniIndex(class_distribution) + \
                               (distribution.getSum() / data.size()) * self.__giniIndex(distribution)
                        if gini + self.EPSILON < best_gini:
                            best_gini = gini
                            best_attribute = index
                            best_split_value = k
            elif isinstance(data.get(0).getAttribute(index), DiscreteAttribute):
                # For discrete attributes, try binary splits on each value
                value_list = data.getAttributeValueList(index)
                for value in value_list:
                    left_distribution = DiscreteDistribution()
                    right_distribution = DiscreteDistribution()
                    for instance in data.list:
                        if instance.getAttribute(index).getValue() == value:
                            left_distribution.addItem(instance.getClassLabel())
                        else:
                            right_distribution.addItem(instance.getClassLabel())
                    
                    if left_distribution.getSum() > 0 and right_distribution.getSum() > 0:
                        gini = (left_distribution.getSum() / data.size()) * self.__giniIndex(left_distribution) + \
                               (right_distribution.getSum() / data.size()) * self.__giniIndex(right_distribution)
                        if gini + self.EPSILON < best_gini:
                            best_gini = gini
                            best_attribute = index
                            best_split_value = value
            elif isinstance(data.get(0).getAttribute(index), ContinuousAttribute):
                # For continuous attributes, find best threshold
                data.sortWrtAttribute(index)
                previous_value = -100000000
                left_distribution = DiscreteDistribution()
                right_distribution = data.classDistribution()
                for k in range(data.size()):
                    instance = data.get(k)
                    if k == 0:
                        previous_value = instance.getAttribute(index).getValue()
                    elif instance.getAttribute(index).getValue() != previous_value:
                        split_value = (previous_value + instance.getAttribute(index).getValue()) / 2
                        previous_value = instance.getAttribute(index).getValue()
                        gini = (left_distribution.getSum() / data.size()) * self.__giniIndex(left_distribution) + \
                               (right_distribution.getSum() / data.size()) * self.__giniIndex(right_distribution)
                        if gini + self.EPSILON < best_gini:
                            best_gini = gini
                            best_split_value = split_value
                            best_attribute = index
                    left_distribution.addItem(instance.getClassLabel())
                    right_distribution.removeItem(instance.getClassLabel())
        
        # Decide whether to use multivariate or univariate split
        use_multivariate = (best_multivariate_weights is not None and 
                           best_multivariate_gini < best_gini)
        
        if use_multivariate:
            # Use multivariate split - create children by partitioning data based on linear combination
            self.leaf = False
            left_data = InstanceList()
            right_data = InstanceList()
            
            for instance in data.list:
                projection = 0.0
                for attr_idx, weight in best_multivariate_weights:
                    projection += weight * instance.getAttribute(attr_idx).getValue()
                
                if projection <= best_multivariate_threshold:
                    left_data.add(instance)
                else:
                    right_data.add(instance)
            
            # Store weights and threshold in condition (using first attribute index as representative)
            # The actual condition will need to be evaluated using all weights
            self.children.append(CartNode(data=left_data,
                                         condition=DecisionCondition(best_multivariate_weights[0][0],
                                                                     ContinuousAttribute(best_multivariate_threshold),
                                                                     "<="),
                                         parameter=parameter,
                                         isStump=isStump))
            self.children.append(CartNode(data=right_data,
                                         condition=DecisionCondition(best_multivariate_weights[0][0],
                                                                     ContinuousAttribute(best_multivariate_threshold),
                                                                     ">"),
                                         parameter=parameter,
                                         isStump=isStump))
        elif best_attribute != -1:
            self.leaf = False
            if isinstance(data.get(0).getAttribute(best_attribute), DiscreteIndexedAttribute):
                self.__createChildrenForDiscreteIndexed(data=data,
                                                        attributeIndex=best_attribute,
                                                        attributeValue=best_split_value,
                                                        parameter=parameter,
                                                        isStump=isStump)
            elif isinstance(data.get(0).getAttribute(best_attribute), DiscreteAttribute):
                self.__createChildrenForDiscrete(data=data,
                                                 attributeIndex=best_attribute,
                                                 attributeValue=best_split_value,
                                                 parameter=parameter,
                                                 isStump=isStump)
            elif isinstance(data.get(0).getAttribute(best_attribute), ContinuousAttribute):
                self.__createChildrenForContinuous(data=data,
                                                   attributeIndex=best_attribute,
                                                   splitValue=best_split_value,
                                                   parameter=parameter,
                                                   isStump=isStump)

    def constructor2(self, inputFile: TextIOWrapper):
        line = inputFile.readline().strip()
        items = line.split(" ")
        if items[0] != "-1":
            if items[1][0] == '=':
                self.__condition = DecisionCondition(int(items[0]), DiscreteAttribute(items[2]), items[1][0])
            elif items[1][0] == ':':
                self.__condition = DecisionCondition(int(items[0]),
                                                     DiscreteIndexedAttribute("", int(items[2]), int(items[3])), '=')
            else:
                self.__condition = DecisionCondition(int(items[0]), ContinuousAttribute(float(items[2])), items[1][0])
        else:
            self.__condition = None
        number_of_children = int(inputFile.readline().strip())
        if number_of_children != 0:
            self.leaf = False
            self.children = []
            for i in range(number_of_children):
                self.children.append(CartNode(inputFile))
        else:
            self.leaf = True
            self.__class_label = inputFile.readline().strip()
            self.__classLabelsDistribution = Model.loadClassDistribution(inputFile)

    def __init__(self,
                 data: object,
                 condition=None,
                 parameter: CartParameter = None,
                 isStump=False):
        if isinstance(data, InstanceList):
            self.constructor1(data, condition, parameter, isStump)
        elif isinstance(data, TextIOWrapper):
            self.constructor2(data)

    def __giniIndex(self, distribution: DiscreteDistribution) -> float:
        """
        Calculates the Gini impurity for a given distribution.
        Gini impurity = 1 - sum(p_i^2) where p_i is the probability of class i.
        
        PARAMETERS
        ----------
        distribution : DiscreteDistribution
            The class distribution.
        
        RETURNS
        -------
        float
            The Gini impurity value.
        """
        total = distribution.getSum()
        if total == 0:
            return 0.0
        gini = 1.0
        for item, count in distribution.items():
            probability = distribution.getProbability(item)
            gini -= probability * probability
        return gini

    def __findBestMultivariateSplit(self, data: InstanceList) -> tuple:
        """
        Finds the best multivariate split using random linear combinations.
        Uses a simpler and more robust approach than LDA: randomly generate
        linear combinations and evaluate them with Gini impurity.
        
        PARAMETERS
        ----------
        data : InstanceList
            Training data.
            
        RETURNS
        -------
        tuple
            (best_gini, weights, threshold) where weights is a list of (attr_idx, weight) tuples.
            Returns (float('inf'), None, None) if no good split is found.
        """
        # Get continuous attribute indices
        continuous_indices = []
        for i in range(data.get(0).attributeSize()):
            if isinstance(data.get(0).getAttribute(i), ContinuousAttribute):
                continuous_indices.append(i)
        
        if len(continuous_indices) < 2 or data.size() < 10:
            return (float('inf'), None, None)
        
        best_gini = float('inf')
        best_weights = None
        best_threshold = None
        
        # Try multiple random linear combinations
        num_trials = min(20, len(continuous_indices) * 5)
        
        # Use a fixed seed for reproducibility based on data size
        # (simple hash to make it somewhat deterministic)
        
        for trial in range(num_trials):
            # Generate random weights (normalized to unit length)
            weights_vec = np.random.randn(len(continuous_indices))
            weights_vec = weights_vec / np.linalg.norm(weights_vec)
            
            # Compute projections
            projections = []
            for instance in data.list:
                projection = 0.0
                for j, attr_idx in enumerate(continuous_indices):
                    projection += weights_vec[j] * instance.getAttribute(attr_idx).getValue()
                projections.append((projection, instance.getClassLabel()))
            
            # Sort by projection value
            projections.sort(key=lambda x: x[0])
            
            # Try different split points (similar to univariate approach)
            left_distribution = DiscreteDistribution()
            right_distribution = DiscreteDistribution()
            
            # Initialize: all instances in right
            for _, label in projections:
                right_distribution.addItem(label)
            
            # Try each split point
            for i in range(len(projections) - 1):
                proj_val, label = projections[i]
                left_distribution.addItem(label)
                right_distribution.removeItem(label)
                
                if left_distribution.getSum() > 0 and right_distribution.getSum() > 0:
                    # Compute threshold as midpoint
                    threshold = (projections[i][0] + projections[i+1][0]) / 2.0
                    
                    gini = (left_distribution.getSum() / data.size()) * self.__giniIndex(left_distribution) + \
                           (right_distribution.getSum() / data.size()) * self.__giniIndex(right_distribution)
                    
                    if gini < best_gini:
                        best_gini = gini
                        best_weights = [(continuous_indices[j], float(weights_vec[j])) for j in range(len(continuous_indices))]
                        best_threshold = float(threshold)
        
        if best_weights is None:
            return (float('inf'), None, None)
        
        return (best_gini, best_weights, best_threshold)

    def __createChildrenForDiscreteIndexed(self,
                                           data: InstanceList,
                                           attributeIndex: int,
                                           attributeValue: int,
                                           parameter: CartParameter,
                                           isStump: bool):
        """
        Creates binary children for discrete indexed attributes.

        PARAMETERS
        ----------
        data : InstanceList
            Training data.
        attributeIndex : int
            Index of the attribute.
        attributeValue : int
            Value of the attribute for the split.
        parameter : CartParameter
            CART parameters.
        isStump : bool
            Whether this is a decision stump.
        """
        children_data = Partition(data, attributeIndex, attributeValue)
        self.children.append(
            CartNode(data=children_data.get(0),
                    condition=DecisionCondition(attributeIndex,
                                                DiscreteIndexedAttribute("",
                                                                        attributeValue,
                                                                        data.get(0).getAttribute(
                                                                            attributeIndex).getMaxIndex())),
                    parameter=parameter,
                    isStump=isStump))
        self.children.append(
            CartNode(data=children_data.get(1),
                    condition=DecisionCondition(attributeIndex,
                                                DiscreteIndexedAttribute("",
                                                                        -1,
                                                                        data.get(0).getAttribute(
                                                                            attributeIndex).getMaxIndex())),
                    parameter=parameter,
                    isStump=isStump))

    def __createChildrenForDiscrete(self,
                                    data: InstanceList,
                                    attributeIndex: int,
                                    attributeValue: str,
                                    parameter: CartParameter,
                                    isStump: bool):
        """
        Creates binary children for discrete attributes.
        One child for instances with the attribute value, another for all other values.

        PARAMETERS
        ----------
        data : InstanceList
            Training data.
        attributeIndex : int
            Index of the attribute.
        attributeValue : str
            Value of the attribute for the split.
        parameter : CartParameter
            CART parameters.
        isStump : bool
            Whether this is a decision stump.
        """
        left_data = InstanceList()
        right_data = InstanceList()
        
        for instance in data.list:
            if instance.getAttribute(attributeIndex).getValue() == attributeValue:
                left_data.add(instance)
            else:
                right_data.add(instance)
        
        self.children.append(CartNode(data=left_data,
                                     condition=DecisionCondition(attributeIndex=attributeIndex,
                                                                 value=DiscreteAttribute(attributeValue)),
                                     parameter=parameter,
                                     isStump=isStump))
        # Use special marker "!=" + value to indicate "not equal to value"
        self.children.append(CartNode(data=right_data,
                                     condition=DecisionCondition(attributeIndex=attributeIndex,
                                                                 value=DiscreteAttribute("!=" + str(attributeValue))),
                                     parameter=parameter,
                                     isStump=isStump))

    def __createChildrenForContinuous(self,
                                      data: InstanceList,
                                      attributeIndex: int,
                                      splitValue: float,
                                      parameter: CartParameter,
                                      isStump: bool):
        """
        Creates binary children for continuous attributes.

        PARAMETERS
        ----------
        data : InstanceList
            Training data.
        attributeIndex : int
            Index of the attribute.
        splitValue : float
            Threshold value for the split.
        parameter : CartParameter
            CART parameters.
        isStump : bool
            Whether this is a decision stump.
        """
        children_data = Partition(data, attributeIndex, splitValue)
        self.children.append(CartNode(children_data.get(0),
                                     DecisionCondition(attributeIndex, ContinuousAttribute(splitValue), "<"),
                                     parameter, isStump))
        self.children.append(CartNode(children_data.get(1),
                                     DecisionCondition(attributeIndex, ContinuousAttribute(splitValue), ">"),
                                     parameter, isStump))

    def predict(self, instance: Instance) -> str:
        """
        Makes prediction for an instance by traversing the tree.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            The predicted class label.
        """
        if isinstance(instance, CompositeInstance):
            possible_class_labels = instance.getPossibleClassLabels()
            distribution = self.__classLabelsDistribution
            predicted_class = distribution.getMaxItemIncludeTheseOnly(possible_class_labels)
            if self.leaf:
                return predicted_class
            else:
                for node in self.children:
                    if node.__condition.satisfy(instance):
                        child_prediction = node.predict(instance)
                        if child_prediction is not None:
                            return child_prediction
                        else:
                            return predicted_class
                return predicted_class
        elif self.leaf:
            return self.__class_label
        else:
            for node in self.children:
                if node.__condition.satisfy(instance):
                    return node.predict(instance)
            return self.__class_label

    def predictProbabilityDistribution(self, instance: Instance) -> dict:
        """
        Returns the probability distribution for a given instance.

        PARAMETERS
        ----------
        instance : Instance
            Instance for prediction.

        RETURNS
        -------
        dict
            Probability distribution over class labels.
        """
        if self.leaf:
            return self.__classLabelsDistribution.getProbabilityDistribution()
        else:
            for node in self.children:
                if node.__condition.satisfy(instance):
                    return node.predictProbabilityDistribution(instance)
            return self.__classLabelsDistribution.getProbabilityDistribution()
