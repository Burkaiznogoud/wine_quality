from sklearn.feature_selection import SelectKBest, f_classif# Select K Best algorithm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from misc.evaluation import evaluation
from misc.timing import timing
from misc.processing import processing


class SKB_Algorithm:
    def __init__(self, k, columns, X_train, Y_train, X_test, Y_test, init_params = 'default'):
        self.parameters =   { 
                            'kernel': ['linear', 'poly'],
                            'C': [0.001, 0.01, 0.1, 1, 10],
                            'gamma':[0.001, 0.01, 0.1, 1, 10],
                            'degree': [3, 4 , 5],
                            'decision_function_shape': ['ovo', 'ovr']
                            }
        self.columns = columns
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.K = k
        self.instantiate_SVC(init_params = init_params)
        self.calculate_Y_hat()
        self.select_features()
        self.evaluate_classification_metrics()
        self.evaluate_regression_metrics()

    @processing
    def instantiate_SVC(self, init_params):
        params =    {  'default':   {  
                                    'kernel': 'linear',
                                    'C' : 1, 
                                    'gamma': 0.1, 
                                    'degree': 3, 
                                    'decision_function_shape': 'ovo'
                                    }
                    }
        if init_params == 'default':
            self.estimator = SVC(**params['default'])
            return self.estimator
        else:
            self.estimator = SVC(**init_params)
            return self.estimator

    @processing
    def calculate_Y_hat(self):
        self.estimator.fit(self.X_train, self.Y_train)
        self.Y_hat = self.estimator.predict(self.X_test)
        return self.Y_hat

    @processing
    def select_features(self):
        skb = SelectKBest(score_func = f_classif, k = self.K)
        skb.fit(self.X_train, self.Y_train)
        selected_indices = skb.get_support(indices=True)
        indices = [i for i in selected_indices]
        self.selected_features = [self.columns[i] for i in selected_indices]
        self.X_test = self.X_test[:, indices]
        self.X_train = self.X_train[:, indices]
        return self.X_train, self.X_test, self.selected_features
    
    @processing
    def evaluate_classification_metrics(self):
        self.accuracy = accuracy_score(self.Y_test, self.Y_hat)
        self.precision = precision_score(self.Y_test, self.Y_hat)
        self.recall = recall_score(self.Y_test, self.Y_hat)
        self.f1 = f1_score(self.Y_test, self.Y_hat)
        return self.accuracy, self.precision, self.recall, self.f1
    
    @processing
    def evaluate_regression_metrics(self):
        self.mae = mean_absolute_error(self.Y_test, self.Y_hat)
        self.mse = mean_squared_error(self.Y_test, self.Y_hat)
        self.r2 = r2_score(self.Y_test, self.Y_hat)
        return self.mae, self.mse, self.r2
    
    @evaluation
    def evaluation_results(self):
        results =   {
                    'results': f" {__name__} of {__class__}",
                    'parameters': f" {self.estimator.get_params}",
                    'feature selection': f" {self.selected_features}",
                    'accuracy': f" {self.accuracy:.4f}",
                    'precision': f" {self.precision:.4f}",
                    'recall': f" {self.recall:.4f}",
                    'f1': f" {self.f1:.4f}",
                    'mae': f" {self.mae:.4f}",
                    'mse': f" {self.mse:.4f}",
                    'r2': f" {self.r2:.4f}"
                    }
        return results
    


"""
The SelectKBest algorithm is a feature selection method in machine learning
that is used to select the top k features based on statistical tests. 
It is part of the scikit-learn library and is particularly useful 
when you want to reduce the dimensionality of your feature space 
by selecting only the most informative features.

Here's an overview of the SelectKBest algorithm, its parameters, and how to use it:

SelectKBest Algorithm:
The SelectKBest algorithm works by scoring each feature 
using a statistical test and selecting the top k features 
based on their scores. Common statistical tests used include:

ANOVA F-statistic: 
This test is suitable for regression problems and measures 
the difference in means of the feature among different output classes.

Mutual Information: 
This non-parametric method measures the dependency 
between two variables and is suitable for both classification and regression problems.

Parameters:
The main parameters of the SelectKBest algorithm include:

score_func: 
The statistical test used to evaluate the features. 
It could be a function from sklearn.feature_selection, 
such as f_classif (ANOVA F-statistic for classification) or 
mutual_info_classif (mutual information for classification).

k: 
The number of top features to select. 
You need to specify the desired number of features you want to retain.

In this example, we use the ANOVA F-statistic (f_classif) as the scoring function. 
You can replace it with other scoring functions based on your problem type (classification or regression). 
The value of k is set to 2, meaning we want to select the top 2 features.

Remember to adjust the parameters based on your specific use case and dataset. 
The SelectKBest algorithm is a valuable tool for improving model efficiency 
and interpretability by selecting the most relevant features.
"""