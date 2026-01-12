from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.DecisionTree.CartNode import CartNode
from Classification.Model.ValidatedModel import ValidatedModel
from Classification.Parameter.CartParameter import CartParameter


class Cart(ValidatedModel):

    __root: CartNode

    def constructor1(self, root: CartNode):
        """
        Constructor that sets root node of the CART decision tree.

        PARAMETERS
        ----------
        root : CartNode
            CartNode type input.
        """
        self.__root = root

    def constructor2(self, fileName: str):
        inputFile = open(fileName, mode='r', encoding='utf-8')
        self.__root = CartNode(inputFile)
        inputFile.close()

    def __init__(self, root: object = None):
        if isinstance(root, CartNode):
            self.constructor1(root)
        elif isinstance(root, str):
            self.constructor2(root)

    def predict(self, instance: Instance) -> str:
        """
        The predict method performs prediction on the root node of given instance, and if it is null, it returns the
        possible class labels. Otherwise it returns the returned class labels.

        PARAMETERS
        ----------
        instance : Instance
            Instance to make prediction.

        RETURNS
        -------
        str
            Possible class labels.
        """
        predicted_class = self.__root.predict(instance)
        if predicted_class is None and isinstance(instance, CompositeInstance):
            predicted_class = instance.getPossibleClassLabels()
        return predicted_class

    def predictProbability(self, instance: Instance) -> dict:
        """
        Calculates the posterior probability distribution for the given instance according to CART model.
        
        PARAMETERS
        ----------
        instance : Instance
            Instance for which posterior probability distribution is calculated.
        
        RETURNS
        -------
        dict
            Posterior probability distribution for the given instance.
        """
        return self.__root.predictProbabilityDistribution(instance)

    def pruneNode(self,
                  node: CartNode,
                  pruneSet: InstanceList):
        """
        The prune method takes a CartNode and an InstanceList as inputs. It checks the classification performance
        of given InstanceList before pruning, i.e making a node leaf, and after pruning. If the after performance is
        better than the before performance it prune the given InstanceList from the tree.

        PARAMETERS
        ----------
        node : CartNode
            CartNode that will be pruned if conditions hold.
        pruneSet : InstanceList
            Small subset of tree that will be removed from tree.
        """
        if node.leaf:
            return
        before = self.testClassifier(pruneSet)
        node.leaf = True
        after = self.testClassifier(pruneSet)
        if after.getAccuracy() < before.getAccuracy():
            node.leaf = False
            for child in node.children:
                self.pruneNode(child, pruneSet)

    def prune(self, pruneSet: InstanceList):
        """
        The prune method takes an InstanceList and performs pruning to the root node.

        PARAMETERS
        ----------
        pruneSet : InstanceList
            InstanceList to perform pruning.
        """
        self.pruneNode(self.__root, pruneSet)

    def train(self,
              trainSet: InstanceList,
              parameters: CartParameter):
        """
        Training algorithm for CART (Classification and Regression Trees) decision tree classifier.
        Uses Gini impurity as the splitting criterion and creates binary decision trees.
        20 percent of the data are left aside for pruning, 80 percent of the data is used for constructing the tree.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm.
        parameters: CartParameter
            Parameter of the CART algorithm.
        """
        if parameters.isPrune():
            partition = Partition(instanceList=trainSet,
                                  ratio=parameters.getCrossValidationRatio(),
                                  seed=parameters.getSeed(),
                                  stratified=True)
            self.constructor1(CartNode(partition.get(1), parameter=parameters))
            self.prune(partition.get(0))
        else:
            self.constructor1(CartNode(trainSet, parameter=parameters))

    def loadModel(self, fileName: str):
        """
        Loads the CART decision tree model from an input file.
        
        PARAMETERS
        ----------
        fileName : str
            File name of the CART decision tree model.
        """
        self.constructor2(fileName)
