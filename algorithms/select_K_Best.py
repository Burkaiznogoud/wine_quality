from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, mutual_info_regression # Select K Best algorithm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score


class SKB_Algorithm:
    def __init__(self, k, columns, X_train, Y_train, X_test, Y_test, init_params = {'kernel': 'poly', 'C': 0.1, 'gamma': 0.1, 'degree': 4, 'decision_function_shape': 'ovr'}):
        self.parameters = { 'kernel': ['linear', 'poly'],
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
        self.estimator = SVC(**init_params)
        self.calculate_Y_hat()
        self.select_features()
        self.evaluate_classification_metrics()
        self.evaluate_regression_metrics()

    def calculate_Y_hat(self):
        self.estimator.fit(self.X_train, self.Y_train)
        self.Y_hat = self.estimator.predict(self.X_test)
        return self.Y_hat

    def select_features(self):
        skb = SelectKBest(score_func = f_classif, k = self.K)
        skb.fit(self.X_train, self.Y_train)
        selected_indices = skb.get_support(indices=True)
        indices = [i for i in selected_indices]
        selected_features = [self.columns[i] for i in selected_indices]
        self.X_test = self.X_test[:, indices]
        self.X_train = self.X_train[:, indices]
        print(f"Selected Features : {selected_features}")
        return self.X_train, self.X_test
    
    def evaluate_classification_metrics(self):
        self.accuracy = accuracy_score(self.Y_test, self.Y_hat)
        self.precision = precision_score(self.Y_test, self.Y_hat)
        self.recall = recall_score(self.Y_test, self.Y_hat)
        self.f1 = f1_score(self.Y_test, self.Y_hat)
        return self.accuracy, self.precision, self.recall, self.f1
    
    def evaluate_regression_metrics(self):
        self.mae = mean_absolute_error(self.Y_test, self.Y_hat)
        self.mse = mean_squared_error(self.Y_test, self.Y_hat)
        self.r2 = r2_score(self.Y_test, self.Y_hat)
        return self.mae, self.mse, self.r2
    
    def evaluation_results(self):
        print("-"*20)
        print(f"Accuracy : {self.accuracy:.4f}")
        print(f"Precison : {self.precision:.4f}")
        print(f"Recall : {self.recall:.4f}")
        print(f"F1 : {self.f1:.4f}")
        print(f"Mean Absolute Error : {self.mae:.4f}")
        print(f"Mean Squared Error : {self.mse:.4f}")
        print(f"R2 Score : {self.r2:.4f}")
        print("-"*20)
    
    
# Example usage:
# Assuming you have your data loaded and instantiated as X_train, Y_train, X_test, and Y_test
# columns = your_column_names
# model = SKB_Algorithm(columns, X_train, Y_train, X_test, Y_test)
# mae, selected_features, skb_estimator = model.evaluate_mean_absolute_error()

# Now you can access mae, selected_features, and skb_estimator for further analysis.


# # Select K Best
# class SKB_Algorithm:
#     def __init__(self, columns, X_train, Y_train, X_test, Y_test):
#         self.parameters = {'k': [3, 5, 7],
#                            'score_func': ['accuracy', 'recall', 'precision', 'mean_squared_error']
#                            }
#         self.columns = columns
#         self.X_train = X_train
#         self.Y_train = Y_train
#         self.X_test = X_test
#         self.Y_test = Y_test
#         self.evaluate_mean_absolute_error()

#     def predict_evaluate(self):
#         estimator = SVC()
#         estimator.fit(self.X_train, self.Y_train)
#         self.Y_hat = estimator.predict(self.X_test)
#         self.accuracy = accuracy_score(self.Y_test, self.Y_hat)
#         self.precision = precision_score(self.Y_test, self.Y_hat)
#         self.recall = recall_score(self.Y_test, self.Y_hat)
#         self.f1 = f1_score(self.Y_test, self.Y_hat)
#         return self.accuracy, self.precision, self.recall, self.f1, self.Y_hat

#     def evaluate_mean_absolute_error(self):
#         print('X'*30)
#         self.estimator = SelectKBest(score_func=mean_absolute_error, k = 3)
#         self.X_train = self.estimator.fit_transform(self.X_train, self.Y_train)
#         print(self.X_train)
#         self.predict_evaluate()
#         self.mae = mean_absolute_error(self.Y_test, self.Y_hat)
#         return self.mae, self.estimator
    
    def evaluate_mean_square_error(self):
        self.estimator = SelectKBest(score_func=mean_squared_error, k = 3)
        self.mse = mean_squared_error(self.Y_test, self.Y_hat)
        return self.mse
    
    def evaluate_mean_r2_score(self):
        self.estimator = SelectKBest(score_func=r2_score, k = 3)
        self.r2 = r2_score(self.Y_test, self.Y_hat)
        return self.r2
    
    # def score_mutual_classif(self):
    #     select_k_best = SelectKBest(score_func = mutual_info_classif,  k = min(self.parameters['k']))
    #     X_train_selected = select_k_best.fit_transform(self.X_train, self.Y_train)
    #     X_test_selected = select_k_best.transform(self.X_test)
    #     clf = RandomForestClassifier()
    #     clf.fit(X_train_selected, self.Y_train)
    #     Y_hat = clf.predict(X_test_selected)
    #     precision = precision_score(self.Y_test, Y_hat)
    #     recall = recall_score(self.Y_test, Y_hat)
    #     accuracy = accuracy_score(self.Y_test, Y_hat)
    #     self.mutual_classification = {  'precision': precision,
    #                                     'recall': recall,
    #                                     'accuracy': accuracy,
    #                                     }
    #     self.estimator = select_k_best
    #     return self.mutual_classification, self.estimator
    
    # def score_mutual_regression(self):
    #     select_k_best = SelectKBest(score_func = mutual_info_regression,  k = min(self.parameters['k']))
    #     X_train_selected = select_k_best.fit_transform(self.X_train, self.Y_train)
    #     X_test_selected = select_k_best.transform(self.X_test)
    #     regressor = RandomForestRegressor(n_estimators=len(self.columns), random_state=4)
    #     regressor.fit(X_train_selected, self.Y_train)
    #     Y_hat = regressor.predict(X_test_selected)
    #     precision = precision_score(self.Y_test, Y_hat)
    #     recall = recall_score(self.Y_test, Y_hat)
    #     accuracy = accuracy_score(self.Y_test, Y_hat)
    #     mse = mean_squared_error(self.Y_test, Y_hat)
    #     self.mutual_regression = {  'precision': precision,
    #                                 'recall': recall,
    #                                 'accuracy': accuracy,
    #                                 'mean_square_error': mse 
    #                                 }
    #     self.estimator = select_k_best
    #     return self.mutual_regression, self.estimator
    

"""
Tuned hyperparameters :(best parameters)  {'k': 3}
Accuracy :  nan
skb = SKB_Algorithm(X_train, X_test, Y_train, Y_test)
skb.hyperparameters_score()

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