from sklearn.dummy import DummyClassifier
from algorithms.algorithm import Algorithm
from misc.processing import processing


class Dummy_Algorithm(Algorithm):
    def __init__(self, columns, X_train, Y_train, X_test, Y_test):
        Algorithm.__init__(self, columns, X_train, Y_train, X_test, Y_test)
        self.instantiate_estimator()
        self.train_classification_metrics()
        self.calculate_Y_hat()
        self.evaluate_classification_metrics()
        self.evaluation_results()
        self.plot_confusion_matrix()

    @processing
    def instantiate_estimator(self):
        self.estimator = DummyClassifier(strategy='most_frequent')
        return self.estimator

