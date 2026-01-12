from Classification.Attribute.Attribute import Attribute
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute
from Classification.Instance.Instance import Instance


class DecisionCondition(object):

    __attribute_index: int
    __comparison: str
    __value: Attribute
    __multivariate_weights: list  # For multivariate splits: list of (attr_idx, weight) tuples
    __multivariate_threshold: float  # Threshold for multivariate split

    def __init__(self,
                 attributeIndex: int,
                 value: Attribute,
                 comparison="=",
                 multivariateWeights: list = None,
                 multivariateThreshold: float = None):
        """
        A constructor that sets attributeIndex and Attribute value. It also assigns equal sign to the comparison
        character.

        PARAMETERS
        ----------
        attributeIndex : int
            Integer number that shows attribute index.
        value : Attribute
            The value of the Attribute.
        comparison : str
            Comparison operator: '=', '!=', '<', '>', '<=', 'multivariate_left', 'multivariate_right'.
        multivariateWeights : list
            For multivariate splits: list of (attr_idx, weight) tuples.
        multivariateThreshold : float
            Threshold for multivariate split.
        """
        self.__attribute_index = attributeIndex
        self.__comparison = comparison
        self.__value = value
        self.__multivariate_weights = multivariateWeights
        self.__multivariate_threshold = multivariateThreshold

    def satisfy(self, instance: Instance):
        """
        The satisfy method takes an Instance as an input.

        If defined Attribute value is a DiscreteIndexedAttribute it compares the index of Attribute of instance at the
        attributeIndex and the index of Attribute value and returns the result.

        If defined Attribute value is a DiscreteAttribute it compares the value of Attribute of instance at the
        attributeIndex and the value of Attribute value and returns the result.

        If defined Attribute value is a ContinuousAttribute it compares the value of Attribute of instance at the
        attributeIndex and the value of Attribute value and returns the result according to the comparison character
        whether it is less than or greater than signs.

        For multivariate splits, computes the linear combination w1*x1 + w2*x2 + ... and compares to threshold.

        PARAMETERS
        ----------
        instance : Instance
            Instance to compare.

        RETURNS
        -------
        bool
            True if given instance satisfies the conditions.
        """
        # Handle multivariate splits
        if self.__comparison == "multivariate_left" and self.__multivariate_weights is not None:
            projection = 0.0
            for attr_idx, weight in self.__multivariate_weights:
                projection += weight * instance.getAttribute(attr_idx).getValue()
            return projection <= self.__multivariate_threshold
        elif self.__comparison == "multivariate_right" and self.__multivariate_weights is not None:
            projection = 0.0
            for attr_idx, weight in self.__multivariate_weights:
                projection += weight * instance.getAttribute(attr_idx).getValue()
            return projection > self.__multivariate_threshold
        
        # Handle univariate splits
        if isinstance(self.__value, DiscreteIndexedAttribute):
            if self.__value.getIndex() != -1:
                return instance.getAttribute(self.__attribute_index).getIndex() == self.__value.getIndex()
            else:
                return True
        elif isinstance(self.__value, DiscreteAttribute):
            # Handle not-equal comparison for discrete attributes
            if self.__comparison == "!=":
                return instance.getAttribute(self.__attribute_index).getValue() != self.__value.getValue()
            else:
                return instance.getAttribute(self.__attribute_index).getValue() == self.__value.getValue()
        elif isinstance(self.__value, ContinuousAttribute):
            if self.__comparison == "<":
                return instance.getAttribute(self.__attribute_index).getValue() <= self.__value.getValue()
            else:
                return instance.getAttribute(self.__attribute_index).getValue() > self.__value.getValue()
        return False
    
    def getMultivariateWeights(self) -> list:
        """
        Returns the multivariate weights if this is a multivariate condition.
        
        RETURNS
        -------
        list
            List of (attr_idx, weight) tuples, or None for univariate conditions.
        """
        return self.__multivariate_weights
    
    def getMultivariateThreshold(self) -> float:
        """
        Returns the multivariate threshold if this is a multivariate condition.
        
        RETURNS
        -------
        float
            Threshold value, or None for univariate conditions.
        """
        return self.__multivariate_threshold
    
    def isMultivariate(self) -> bool:
        """
        Checks if this is a multivariate condition.
        
        RETURNS
        -------
        bool
            True if multivariate, False otherwise.
        """
        return self.__multivariate_weights is not None
