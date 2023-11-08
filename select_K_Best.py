from sklearn.feature_selection import SelectKBest, f_classif # Select K Best algorithm
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Select K Best
class SKB_Algorithm:
    def __init__(self):
        self.parameters = {'k': [3, 5, 7]}
        self.scorers = {'precision_score': make_scorer(precision_score),
                        'recall_score': make_scorer(recall_score),
                        'accuracy_score': make_scorer(accuracy_score)
                        }
        self.estimator = SelectKBest(score_func=f_classif)

"""
Tuned hyperparameters :(best parameters)  {'k': 3}
Accuracy :  nan
skb = SKB_Algorithm(X_train, X_test, Y_train, Y_test)
skb.hyperparameters_score()
"""