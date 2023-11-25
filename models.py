
from algorithms import select_K_Best as skb
from algorithms import svm
from algorithms import recursive_feature_elimination as rfe
from algorithms import logistic_regression as lr
from algorithms import decision_tree as dt
from algorithms import naive_algorithm as naive
import warnings

warnings.filterwarnings('ignore')

### Dummy Algorithm ### TESTED and WORKS!
n = naive.Dummy_Algorithm()       

### Select K Best ### TESTED and WORKS!
select_k_best = skb.SKB_Algorithm()

### Recursive Feature Elimination ### TESTED and WORKS but further ivestigation is needed.
recursive_selection = rfe.RFE_Algorithm()

### Logistic Regression ### TESTED and WORKS but further investigation is needed.
logistic = lr.LogisticRegression_Algorithm()

### Decision Tree Classifier ### TESTED and WORKS. Not sure about results.
decision_tree = dt.DecisionTree_Algorithm()
decision_tree.plot_decision_tree()

### Support Vector Machine ### TESTED and WORKS but zero-division error pops up! Try to handle this warning and understand its source.
supp_vector = svm.SVM_Algorithm()


