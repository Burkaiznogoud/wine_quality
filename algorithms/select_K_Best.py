from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from algorithms.algorithm import Algorithm
from misc.evaluation import evaluation
from misc.processing import processing


class SKB_Algorithm(Algorithm):
    def __init__(self, k, columns, X_train, Y_train, X_test, Y_test, init_params = 'default'):
        Algorithm.__init__(columns, X_train, Y_train, X_test, Y_test)
        self.parameters =   { 
                            'kernel': ['linear', 'poly'],
                            'C': [0.001, 0.01, 0.1, 1, 10],
                            'gamma':[0.001, 0.01, 0.1, 1, 10],
                            'degree': [3, 4 , 5],
                            'decision_function_shape': ['ovo', 'ovr']
                            }
        self.K = k
        self.instantiate_SVC(init_params = init_params)
        self.train_classification_metrics()
        self.calculate_Y_hat()
        self.select_features()
        self.evaluate_classification_metrics()

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
    def select_features(self):
        skb = SelectKBest(score_func = f_classif, k = self.K)
        skb.fit(self.X_train, self.Y_train)
        selected_indices = skb.get_support(indices=True)
        indices = [i for i in selected_indices]
        self.selected_features = [self.columns[i] for i in selected_indices]
        self.X_test = self.X_test[:, indices]
        self.X_train = self.X_train[:, indices]
        return self.X_train, self.X_test, self.selected_features
    
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
                    'classification report': f"{self.classification_report:.4f}"
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