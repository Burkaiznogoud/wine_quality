from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from algorithms.algorithm import Algorithm
from misc.evaluation import evaluation
from misc.processing import processing

# Recursive Feature Elimination
class RFE_Algorithm(Algorithm):
    def __init__(self, init_params = 'default'):
        Algorithm.__init__
        self.assign_data()
        self.parameters =   {
                            'n_features_to_select': [2, 4, 6], 'step': [2, 3, 4]
                            }
        self.instantiate_LR()
        self.calculate_Y_hat()
        self.instantiate_RFE(init_params = init_params)
        self.train_classification_metrics()
        self.evaluate_classification_metrics()
        self.results_()
        self.feature_selection()
        self.show_results()

    @processing
    def instantiate_LR(self):
        init_params = {"C": 0.1,'penalty': 'l2', 'solver': 'lbfgs'}
        self.lr = LogisticRegression(**init_params)
        return self.lr

    @processing
    def calculate_Y_hat(self):
        self.lr.fit(self.X_train, self.Y_train)
        self.Y_hat = self.lr.predict(self.X_test)
        return self.Y_hat, self.lr
    
    @processing
    def instantiate_RFE(self, init_params):
        params =    {  
                    'default':  {'n_features_to_select': 5, 'step': 1}
                    }
        if init_params == 'default':
            self.estimator = RFE(estimator = self.lr, **params['default'])
            return self.estimator
        else:
            self.estimator = RFE(estimator = self.lr, **init_params)
            return self.estimator
    
    @processing
    def feature_selection(self):
        self.estimator.fit(self.X_train, self.Y_train)
        features = {feature: i for i, feature in enumerate(self.columns) if self.estimator.support_[i]}
        indices = [i for i, selected in enumerate(self.estimator.support_) if selected]
        self.selected_features = {k: v for k, v in sorted(features.items(), key = lambda item: item[1], reverse = True)}
        self.X_train = self.X_train[:, indices]
        self.X_test = self.X_test[:, indices]
        update_results = {'feature ranking': f" {self.estimator.ranking_}", 'feature selection': f" {self.selected_features}"}
        self.results.update(update_results)
        return self.estimator.ranking_, self.selected_features, self.X_train, self.X_test, self.estimator
    
    @evaluation
    def show_results(self):
        return self.results

"""
Recursive Feature Elimination (RFE) is a feature selection algorithm used in machine learning
to systematically eliminate less important features from a dataset.
It is typically applied in the context of supervised learning,
where you have a labeled dataset and want to identify the most relevant features for your predictive model. 
RFE works by iteratively training the model on subsets of features and eliminating the least important ones, 
continuing this process until a desired number of features or a predefined stopping criterion is met.

Here's an overview of how RFE works and its key parameters:

Input Data: You start with a dataset containing features and their corresponding labels. 
The goal is to select the most important features to improve model performance or reduce dimensionality.

Algorithm Steps:

Initialize: RFE starts with all features included.
Model Training: A machine learning model (e.g., logistic regression, support vector machine, random forest) is trained on the entire feature set.
Feature Ranking: After training, the importance of each feature is assessed, 
usually by examining their coefficients or feature importances.
Feature Elimination: The least important feature(s) is removed from the dataset.
Iteration: Steps 2-4 are repeated until a specified number of features is reached, 
or a predefined stopping criterion (e.g., a target performance metric) is met.

Parameters:

estimator: This parameter specifies the machine learning algorithm to be used for feature ranking and selection. 
You can choose a classifier or regressor depending on your problem type.

n_features_to_select: It specifies the desired number of features to be retained after the RFE process. 
If not specified, RFE will select the top 50% of features by default.

step: This parameter determines how many features are removed at each iteration. 
A lower value of step means that fewer features are eliminated in each iteration, resulting in a more selective process. 
The default value is 1, meaning one feature is removed at each step.

verbose: Controls the amount of information printed during the RFE process. 
A higher value provides more detailed information about each iteration.

scoring: Specifies the evaluation metric used to rank the features. 
It can be a scoring function that measures the quality of the model 
(e.g., accuracy, F1-score, mean squared error) or a custom function.

Adjusting Parameters:

You can adjust the n_features_to_select parameter to control the final number of features you want in your model. 
Set it to the desired number or use techniques like cross-validation to find an optimal number.

The step parameter can be adjusted to change the aggressiveness of feature elimination. 
A smaller step will result in a more gradual elimination process, while a larger step will be more aggressive.

The scoring parameter should be chosen based on the problem you are solving. 
You may need to experiment with different scoring functions to find the one that best reflects your model's performance.

RFE is a valuable technique for feature selection, as it helps improve model interpretability, reduce overfitting,
and often leads to better model performance by focusing on the most informative features.
However, it can be computationally expensive for large datasets with many features,
so it's essential to balance the trade-off between computational cost and feature selection accuracy.
"""