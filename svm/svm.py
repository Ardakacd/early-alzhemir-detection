import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score

class OneVsAllSVM:
    def __init__(self, n_components=2):
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    def fit(self, X_train, y_train):
        # Standardize the data
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train

        self.classes = np.unique(y_train)
        self.best_params_ = {}

        for cls in self.classes:
            y_binary = np.where(y_train == cls, 1, -1)
            C = 1e2
            print("C is",C)
            model = SVC(kernel='linear', C=C)
            model.fit(self.X_train, y_binary)
            self.models[cls] = model

        # Fit PCA for visualization
        self.X_train_pca = self.pca.fit_transform(self.X_train)

    def predict(self, X_test):
        # Standardize the test data
        X_test = self.scaler.transform(X_test)
        self.X_test_pca = self.pca.transform(X_test)

        # Compute decision function for each model
        decision_function = np.zeros((X_test.shape[0], len(self.classes)))
        for i, cls in enumerate(self.classes):
            decision_function[:, i] = self.models[cls].decision_function(X_test)

        # Assign each sample to the class with the highest decision function value
        return self.classes[np.argmax(decision_function, axis=1)]

    def visualize(self):
        for i, cls in enumerate(self.classes):
            plt.figure()
            plt.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1], c=self.y_train == cls, cmap='viridis', alpha=0.6)
            w = self.models[cls].coef_[0]
            b = self.models[cls].intercept_[0]
            x_plot = np.linspace(min(self.X_train_pca[:, 0]), max(self.X_train_pca[:, 0]), 200)
            y_plot = -(w[0] / w[1]) * x_plot - b / w[1]
            plt.plot(x_plot, y_plot, 'r')
            plt.title(f"Class {cls} vs All")
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend([f'Margin {cls}'], loc='best')
            plt.show()

    def report(self, X_test, y_test):
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=[f'Class {cls}' for cls in self.classes])
        print(report)
        return report


