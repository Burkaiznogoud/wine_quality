from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from misc.evaluation import evaluation
from misc.timing import timing
from misc.processing import processing

# Decision Tree Classifier
class DecisionTree_Algorithm:
    def __init__(self, X_train, Y_train, X_test, Y_test, columns, init_params = 'default'):
        self.parameters = { 'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [4, 5, 6],
                            'max_features': [None, 'sqrt', 'log2', 5, 7],
                            'min_samples_leaf': [4, 8, 16],
                            'min_samples_split': [4, 8, 16]}
        self.instantiate_DT(init_params = init_params)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.columns = columns
        self.train_classification_metrics()
        self.calculate_Y_hat()
        self.feature_selection()
        self.evaluate_classification_metrics()
        self.evaluation_results()

    @processing
    def instantiate_DT(self, init_params):
        params = {'default':    {  
                                'criterion': 'entropy',
                                'splitter': 'random',
                                'max_depth': 5,
                                'max_features': 'sqrt',
                                'min_samples_leaf': 8,
                                'min_samples_split': 8,
                                'ccp_alpha': 0.1,
                                'random_state': 40
                                }
                }
        if init_params == 'default':
            self.estimator = DecisionTreeClassifier(**params['default'])
            return self.estimator
        else:
            self.estimator = DecisionTreeClassifier(**init_params)
            return self.estimator

    @evaluation
    def train_classification_metrics(self):
        self.estimator.fit(self.X_train, self.Y_train)
        Y_hat = self.estimator.predict(self.X_train)
        accuracy = accuracy_score(self.Y_train, Y_hat)
        precision = precision_score(self.Y_train, Y_hat)
        recall = recall_score(self.Y_train, Y_hat)
        f1 = f1_score(self.Y_train, Y_hat)
        classif_report = classification_report(self.Y_train, Y_hat)
        self.train_report = {
            'test accuracy': f"{accuracy:.4f}",
            'test precision': f"{precision:.4f}",
            'test recall': f"{recall:.4f}",
            'test f1': f"{f1:.4f}",
            'test classification report': f"{classif_report}",
        }
        return self.train_report

    @processing
    def calculate_Y_hat(self):
        self.estimator.fit(self.X_train, self.Y_train)
        self.Y_hat = self.estimator.predict(self.X_test)
        return self.Y_hat

    @processing
    def evaluate_classification_metrics(self):
        self.accuracy = accuracy_score(self.Y_test, self.Y_hat)
        self.precision = precision_score(self.Y_test, self.Y_hat)
        self.recall = recall_score(self.Y_test, self.Y_hat)
        self.f1 = f1_score(self.Y_test, self.Y_hat)
        self.classification_report = classification_report(self.Y_test, self.Y_hat)
        return self.accuracy, self.precision, self.recall, self.f1, self.classification_report

    @processing
    def feature_selection(self):
        self.estimator.fit(self.X_train, self.Y_train)
        feature_importances = self.estimator.feature_importances_
        feature_names = self.columns
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        self.features_selected = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        return self.features_selected

    @evaluation
    def evaluation_results(self):
        results =   {
                    'results': f"{__name__} of {__class__}",
                    'parameters': f"{self.estimator.get_params}",
                    'feature selection': f" {self.features_selected}",
                    'accuracy': f"{self.accuracy:.4f}",
                    'precision': f"{self.precision:.4f}",
                    'recall': f"{self.recall:.4f}",
                    'f1': f"{self.f1:.4f}",
                    'classification report': f"{self.classification_report:.4f}"
                    }
        return results

"""
The Decision Tree algorithm is a popular machine learning algorithm used for both classification and regression tasks. 
It's a non-parametric supervised learning method that is simple to understand, interpret, and visualize. 
A Decision Tree works by recursively splitting the dataset into subsets based on the most significant attribute at each node, 
aiming to create homogeneous leaf nodes that represent the target variable.

Key Features of the Decision Tree Algorithm:

1. **Non-Linearity**: Decision Trees can model complex, non-linear relationships in the data.

2. **Interpretability**: The resulting tree structure is easy to interpret and explain, 
making it useful for decision support and understanding the model's logic.

3. **Variable Selection**: Decision Trees can automatically select
the most relevant features for the problem by identifying important splits early in the tree.

4. **Handling Mixed Data**: Decision Trees can handle both categorical and numerical features.

5. **Robustness**: They are resistant to outliers and can work with noisy data.

6. **Ensemble Learning**: Decision Trees are often used as base learners in ensemble methods like Random Forests and Gradient Boosting.

Parameters to Tune for Best Model Fitting:

1. **Criterion (criterion)**: 
Decision Trees can be built using different criteria for measuring the quality of splits, 
such as "gini" for the Gini impurity, or "entropy" for information gain. 
The choice of criterion depends on the problem and data.

2. **Max Depth (max_depth)**: 
This parameter controls the maximum depth of the tree. 
A deeper tree may lead to overfitting, so it's essential to tune this parameter carefully.

3. **Min Samples Split (min_samples_split)**: 
It specifies the minimum number of samples required to split an internal node. 
Setting it to a higher value can help avoid overfitting.

4. **Min Samples Leaf (min_samples_leaf)**: 
It sets the minimum number of samples required in a leaf node. 
It can be used to control the granularity of the tree.

5. **Max Features (max_features)**: 
It determines the maximum number of features to consider when making a split. 
Setting it can help improve model performance and reduce overfitting.

6. **Splitter (splitter)**: 
The choice of the splitting strategy, which can be "best" or "random." 
"Best" selects the best split, while "random" selects the best random split. 
"Best" is usually a safe choice, but "random" can help in reducing overfitting.

7. **Class Weight (class_weight)**: 
You can assign weights to classes to address class imbalance problems.

8. **Pruning Parameters**: 
Some Decision Tree implementations may provide pruning parameters 
to control the tree's size and complexity, like "min_impurity_decrease."

9. **Randomness Parameters**: 
For randomized Decision Trees, you might have parameters 
to control the randomness of attribute selection (e.g., "max_features" and "random_state").

10. **Regularization Parameters**: 
Some implementations might offer regularization techniques like cost-complexity pruning.

To find the best set of hyperparameters for your Decision Tree model, 
you can use techniques like grid search or random search with cross-validation 
to evaluate different combinations of these parameters and select the one 
that results in the best model performance. 
Additionally, you should keep an eye on the depth of the tree 
to avoid overfitting and consider the use of ensemble methods 
like Random Forest to further improve the model's predictive power and robustness.
""" 