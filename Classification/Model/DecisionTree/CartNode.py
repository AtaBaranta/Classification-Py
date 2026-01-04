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
        the Gini impurity as the splitting criterion (CART algorithm). It finds the best attribute to split on
        by evaluating all attributes and choosing the one with the lowest Gini impurity.

        CART always creates binary splits:
        - For continuous attributes: split at a threshold value
        - For discrete attributes: create a binary split based on equality to a value
        - For discrete indexed attributes: create a binary split

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
        
        if best_attribute != -1:
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
