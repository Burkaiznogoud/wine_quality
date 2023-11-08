from sklearn.model_selection import GridSearchCV # Allows us to test parameters of classification algorithms and find the best one
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Algorithm:
    def __init__(self, _X_train, _X_test, _Y_train, _Y_test, estimator, params):
        self.parameters = params
        self.X = _X_train
        self.Y = _Y_train
        self.x = _X_test
        self.y = _Y_test
        self.estimator = estimator
        self.cv = GridSearchCV(cv = 10, param_grid = self.parameters, estimator = self.estimator)
        self.fit, self.score, self.Y_hat = self.fit_score_predict()

    def fit_score_predict(self):
        self.fit = self.cv.fit(self.X, self.Y)
        self.score = self.cv.score(self.x, self.y)
        self.Y_hat = self.cv.predict(self.x)
        return self.fit, self.score, self.Y_hat

    def hyperparameters_score(self):
        print("Tuned hyperparameters :(best parameters) ", self.cv.best_params_)
        print("Accuracy : ", self.cv.best_score_)
        print(self.score)

    def plot_confusion_matrix(self):
        "this function plots the confusion matrix"
        cm = confusion_matrix(self.y, self.Y_hat)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Not recommended', 'Recommended']); ax.yaxis.set_ticklabels(['Not recommended', 'Recommended'])
        plt.show()