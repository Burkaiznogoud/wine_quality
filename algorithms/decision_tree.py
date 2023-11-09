from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier algorithm

# Decision Tree Classifier
class DecisionTree_Algorithm:
    def __init__(self):
        self.parameters = { 'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [2, 3, 4],
                            'max_features': [None, 'sqrt', 'log2', 0.5, 10],
                            'min_samples_leaf': [2, 4, 8],
                            'min_samples_split': [2, 4, 8]}
        self.estimator = DecisionTreeClassifier(ccp_alpha=0.01, random_state=4)

    def get_feature_importance(self, X, Y, columns):
        self.estimator.fit(X, Y)
        feature_importances = self.estimator.feature_importances_
        feature_names = columns
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        self.important_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        return self.important_features

"""Tuned hyperparameters :(best parameters)  {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
Accuracy :  0.9989010989010989
Score: 1.0
tree = DecisionTree_Algorithm(X_train, X_test, Y_train, Y_test)
tree.hyperparameters_score()

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