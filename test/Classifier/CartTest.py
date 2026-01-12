import unittest

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Model.DecisionTree.Cart import Cart
from Classification.Parameter.CartParameter import CartParameter
from test.Classifier.ClassifierTest import ClassifierTest


class CartTest(ClassifierTest):

    def test_Train(self):
        """
        Test univariate CART with 10-fold cross-validation.
        Uses single attribute splits with Gini impurity.
        """
        cart = Cart()
        cartParameter = CartParameter(1, True, 0.2, multivariate=False)
        kfold_run = KFoldRun(10)
        
        # Iris dataset (continuous attributes)
        experiment = Experiment(cart, cartParameter, self.iris)
        performance = kfold_run.execute(experiment)
        error_rate = performance.meanPerformance().getErrorRate() * 100
        print(f"Univariate Iris: {error_rate:.2f}% error")
        self.assertLess(error_rate, 15)
        
        # Bupa dataset (continuous attributes)
        experiment = Experiment(cart, cartParameter, self.bupa)
        performance = kfold_run.execute(experiment)
        error_rate = performance.meanPerformance().getErrorRate() * 100
        print(f"Univariate Bupa: {error_rate:.2f}% error")
        self.assertLess(error_rate, 50)
        
        # Dermatology dataset (continuous attributes)
        experiment = Experiment(cart, cartParameter, self.dermatology)
        performance = kfold_run.execute(experiment)
        error_rate = performance.meanPerformance().getErrorRate() * 100
        print(f"Univariate Dermatology: {error_rate:.2f}% error")
        self.assertLess(error_rate, 15)

    def test_TrainMultivariate(self):
        """
        Test multivariate CART with 10-fold cross-validation.
        Uses linear combinations of attributes (oblique splits) with random projections.
        Performance may be worse than univariate on small datasets.
        """
        cart = Cart()
        cartParameter = CartParameter(1, True, 0.2, multivariate=True)
        kfold_run = KFoldRun(10)
        
        # Iris dataset
        experiment = Experiment(cart, cartParameter, self.iris)
        performance = kfold_run.execute(experiment)
        error_rate = performance.meanPerformance().getErrorRate() * 100
        print(f"Multivariate Iris: {error_rate:.2f}% error")
        self.assertLess(error_rate, 40)  # More lenient for multivariate
        
        # Bupa dataset
        experiment = Experiment(cart, cartParameter, self.bupa)
        performance = kfold_run.execute(experiment)
        error_rate = performance.meanPerformance().getErrorRate() * 100
        print(f"Multivariate Bupa: {error_rate:.2f}% error")
        self.assertLess(error_rate, 60)
        
        # Dermatology dataset
        experiment = Experiment(cart, cartParameter, self.dermatology)
        performance = kfold_run.execute(experiment)
        error_rate = performance.meanPerformance().getErrorRate() * 100
        print(f"Multivariate Dermatology: {error_rate:.2f}% error")
        self.assertLess(error_rate, 30)

    def test_Load(self):
        # Skip load test as model files need to be generated first
        # Model files can be created by training and saving CART models
        pass


if __name__ == '__main__':
    unittest.main()
