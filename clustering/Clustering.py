import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class Clustering:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
    
    def fit(self, X_train, X_test, y_train, y_test):
        # Standardize the training and test data
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Fit the KMeans model on the training data
        self.model.fit(self.X_train_scaled)
        
        # Predict the clusters for both training and test data
        self.train_labels_ = self.model.predict(self.X_train_scaled)
        self.test_labels_ = self.model.predict(self.X_test_scaled)
        
        # Fit PCA for visualization on the training data
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)

        # Generate classification reports
        self.train_classification_report = classification_report(y_train, self.train_labels_, target_names=['Normal cognition', 'MCI', 'Dementia'])
        self.test_classification_report = classification_report(y_test, self.test_labels_, target_names=['Normal cognition',  'MCI', 'Dementia'])
    
    def visualize(self, dataset='train'):
        if dataset == 'train':
            X_pca = self.X_train_pca
            labels = self.train_labels_
            title = 'Training Data Clusters Visualization'
        elif dataset == 'test':
            X_pca = self.X_test_pca
            labels = self.test_labels_
            title = 'Test Data Clusters Visualization'
        else:
            raise ValueError("Dataset parameter should be 'train' or 'test'")
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Create a legend
        legend_labels = ['Normal cognition', 'Impaired-not-MCI', 'MCI', 'Dementia']
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/4), markersize=10) for i in range(4)]
        plt.legend(handles, legend_labels, title="Labels")
        
        plt.colorbar(scatter, ticks=range(self.n_clusters), label='Cluster Label')
        plt.show()
    
    def print_classification_report(self, dataset='train'):
        if dataset == 'train':
            print(self.train_classification_report)
        elif dataset == 'test':
            print(self.test_classification_report)
        else:
            raise ValueError("Dataset parameter should be 'train' or 'test'")



    
    
