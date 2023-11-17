from sklearn.model_selection import GridSearchCV # Allows us to test parameters of classification algorithms and find the best one
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from misc.timing import timing
from misc.evaluation import evaluation
from misc.processing import processing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import warnings


class Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test, params, estimator):
        self.parameters = params
        self.estimator = estimator
        self.scoring =  {
                        'accuracy': make_scorer(accuracy_score), 
                        'precision': make_scorer(precision_score), 
                        'recall': make_scorer(recall_score), 
                        'f1': make_scorer(f1_score)
                        }
        self.X = _X_train
        self.Y = _Y_train
        self.x = _X_test
        self.y = _Y_test
        self.cv = GridSearchCV(estimator = self.estimator, 
                               cv = 10, 
                               param_grid = self.parameters, 
                               scoring = self.scoring,
                               refit = 'accuracy',
                               n_jobs = 2
                               )
        self.fit_score_predict()
        self.calculate_hyperparameters()
        self.hyperparameters_info()

    @processing
    def fit_score_predict(self):
        warnings.filterwarnings('ignore')
        self.fit = self.cv.fit(self.X, self.Y)
        self.score = self.cv.score(self.x, self.y)
        self.Y_hat = self.cv.predict(self.x)
        return self.fit, self.score, self.Y_hat

    @processing
    def calculate_hyperparameters(self):
        self.best_params = self.cv.best_params_
        self.best_score = self.score
        self.best_accuracy = self.cv.best_score_
        return self.best_params, self.best_accuracy, self.best_score
    
    @evaluation
    def hyperparameters_info(self):
        results = {
            'best params': f" {self.best_params}",
            'best score': f" {self.best_score}",
            'best accuracy': f" {self.best_accuracy}"
        }
        return results

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y, self.Y_hat)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax, fmt='d')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Not recommended', 'Recommended']) 
        ax.yaxis.set_ticklabels(['Not recommended', 'Recommended'])
        plt.show()