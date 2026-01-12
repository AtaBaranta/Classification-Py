from Classification.Parameter.Parameter import Parameter


class CartParameter(Parameter):

    __prune: bool
    __cross_validation_ratio: float
    __multivariate: bool

    def __init__(self,
                 seed: int,
                 prune: bool,
                 crossValidationRatio: float,
                 multivariate: bool = False):
        """
        Parameters of the CART (Classification and Regression Trees) decision tree classifier.
        Supports both univariate (single attribute) and multivariate (linear combinations) splits.

        PARAMETERS
        ----------
        seed : int
            Seed is used for random number generation.
        prune : bool
            Boolean value for prune.
        crossValidationRatio : float
            Double value for cross validation ratio.
        multivariate : bool
            If True, uses multivariate splits (linear combinations of attributes).
            If False, uses univariate splits (single attribute per split). Default: False.
        """
        super().__init__(seed)
        self.__prune = prune
        self.__cross_validation_ratio = crossValidationRatio
        self.__multivariate = multivariate

    def isPrune(self) -> bool:
        """
        Accessor for the prune.

        RETURNS
        -------
        bool
            Prune.
        """
        return self.__prune

    def getCrossValidationRatio(self) -> float:
        """
        Accessor for the crossValidationRatio.

        RETURNS
        -------
        float
            crossValidationRatio.
        """
        return self.__cross_validation_ratio

    def isMultivariate(self) -> bool:
        """
        Accessor for the multivariate flag.

        RETURNS
        -------
        bool
            True if using multivariate splits, False for univariate.
        """
        return self.__multivariate
