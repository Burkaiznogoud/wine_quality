from sklearn.linear_model import LogisticRegression
from algorithms.algorithm import Algorithm
from misc.processing import processing
from misc.evaluation import evaluation

class LogisticRegression_Algorithm(Algorithm):
    def __init__(self, init_params = 'default'):
        Algorithm.__init__
        self.assign_data()
        self.parameters =   {
                            'penalty':['l2'],
                            'C': [0.1, 1, 10],
                            'fit_intercept': [True, False],
                            'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
                            'max_iter': [500, 1000],
                            'multi_class': ['ovr', 'multinomial'],
                            'random_state': [4, 8, 16],
                            'tol': [1e-2, 1e-3, 1e-4]
                            }
        self.instantiate_LR(init_params = init_params)
        self.train_classification_metrics()
        self.calculate_Y_hat()
        self.evaluate_classification_metrics()
        self.results_()
        self.feature_selection()
        self.show_results()

    @processing
    def instantiate_LR(self, init_params):
        params = {'default': {  'penalty': 'l2',
                                'C' : 1, 
                                'fit_intercept': True, 
                                'solver': 'lbfgs', 
                                'max_iter': 800,
                                'multi_class': 'ovr',
                                'random_state': 8,
                                'tol': 0.1}
                  }
        if init_params == 'default':
            self.estimator = LogisticRegression(**params['default'])
            return self.estimator
        else:
            self.estimator = LogisticRegression(**init_params)
            return self.estimator

    @evaluation        
    def feature_selection(self):
        coefficients = self.estimator.coef_[0]
        self.features_selected = {feature: coeff for feature, coeff in zip(self.columns, coefficients)}
        return self.features_selected
        
        
"""
Logistic Regression Algorithm:
Logistic Regression is a binary classification algorithm
used for predicting the probability that an instance belongs to a particular category. 
Despite its name, it is used for classification, not regression. 
The logistic regression model is based on the logistic function 
(also called the sigmoid function), which has an S-shaped curve. 
 
f(x) = 1 / 1 + e^(-x)

In logistic regression,
the input features are linearly combined with weights, 
and the result is passed through the logistic function. 
Mathematically, for a binary classification problem with features 
x1, x2, ... , xn and weights w1, w2, ... , wn, 
the logistic regression model can be represented as:

P( Y = 1 ) = 1 / 1 + e^-(w0 + w1x1 + w2x2 + .. + wnxn)

where 

P( Y = 1 ) is the probability of the instance belonging to class 1, 
and Y is the binary output (0 or 1).

The logistic regression model is trained by adjusting 
the weights to minimize a cost function, 
typically the cross-entropy loss. 
This is often done using optimization algorithms like gradient descent.

Evaluation of Logistic Regression:

Accuracy: 
The proportion of correctly classified instances out of the total instances. 
It is calculated as:

ACC = Number of Correct Predictions / Total Number of Predictions

Precision: 
The proportion of true positive predictions out of the total predicted positives. 
It is calculated as:

PREC = True Positives / ( True Positives + False Positives ) 

Recall (Sensitivity or True Positive Rate): 
The proportion of true positive predictions out of the total actual positives. 
It is calculated as:

REC = True Positives / ( True Positives + False Negatives )

F1 Score: 
The harmonic mean of precision and recall. 
It balances precision and recall. It is calculated as:

F1 Score = (2 x Precision x Recall) / ( Precision + Recall )

ROC Curve and AUC: 
Receiver Operating Characteristic (ROC) curve 
is a graphical representation of the model's ability 
to distinguish between classes. 
The Area Under the Curve (AUC) summarizes the ROC curve, 
with higher values indicating better performance.

Confusion Matrix: 
A table that shows the true positive, true negative, 
false positive, and false negative predictions, 
providing an overall view of the model's performance.

When evaluating a logistic regression model, 
the choice of metrics depends on the specific goals and requirements of the task. 
For example, in a medical diagnosis scenario, 
recall might be more critical to minimize false negatives, 
even at the cost of increased false positives.

Parametrization:
enalty (default='l2'):

Specifies the norm used in the regularization term. It can be 'l1', 'l2', 'elasticnet', or 'none'.
C (default=1.0):

Inverse of regularization strength. Smaller values specify stronger regularization.
fit_intercept (default=True):

Specifies whether to fit an intercept term (bias).
solver (default='lbfgs'):

Algorithm to use in the optimization problem. Common choices include 'newton-cg', 'lbfgs', 'liblinear', 'sag', and 'saga'.
max_iter (default=100):

Maximum number of iterations for optimization algorithms to converge.
multi_class (default='auto'):

Specifies how to handle multiple classes. Options include 'ovr' (one-vs-rest) and 'multinomial' (softmax).
dual (default=False):

Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
class_weight (default=None):

Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
random_state (default=None):

Seed of the pseudo-random number generator.
tol (default=1e-4):
"""
