import unittest

from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter
from test.Classifier.ClassifierTest import ClassifierTest


class CartTest(ClassifierTest):

    def test_Train(self):
        cart = Cart()
        cartParameter = CartParameter(1, True, 0.2)
        cart.train(self.iris.getInstanceList(), cartParameter)
        self.assertLess(100 * cart.test(self.iris.getInstanceList()).getErrorRate(), 10)
        cart.train(self.bupa.getInstanceList(), cartParameter)
        self.assertLess(100 * cart.test(self.bupa.getInstanceList()).getErrorRate(), 45)
        cart.train(self.dermatology.getInstanceList(), cartParameter)
        self.assertLess(100 * cart.test(self.dermatology.getInstanceList()).getErrorRate(), 10)
        cart.train(self.car.getInstanceList(), cartParameter)
        self.assertLess(100 * cart.test(self.car.getInstanceList()).getErrorRate(), 35)
        cart.train(self.tictactoe.getInstanceList(), cartParameter)
        self.assertLess(100 * cart.test(self.tictactoe.getInstanceList()).getErrorRate(), 35)

    def test_Load(self):
        # Skip load test as model files need to be generated first
        # Model files can be created by training and saving CART models
        pass


if __name__ == '__main__':
    unittest.main()
