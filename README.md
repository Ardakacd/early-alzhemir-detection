# Early Alzheimer Detection with Multiple Modalities

### Introduction
In this project, we delve into the realm of multi-modal representation learning, focusing on its application in healthcare. Collaborating closely, we navigate through the complexities of training deep models, extracting word embeddings, and preprocessing diverse tabular data types, all essential components for tackling the complexities of early Alzheimer's detection. As we progress, our journey will result in the design of a sophisticated deep network adept at seamlessly integrating representations from multiple modalities.

### Project Overview
#### Until Interim Report
In the early stages of our project we tried to apply word embeddings on the tabular data we are working on and observe whether we can come up with more accurate deep models. Before starting to work on real-world data that we are planning to source from NACC (National Alzheimer’s Coordinating Center), we started to work on a publicly available [dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) that has similar properties as NACC data. Throughout our process until interim report we may list the topics we have learned and implemented as follows:

* What does "word embedding" mean ?
* What are some tecniques/algorithms used in word embeddings (e.g. word2vec)?
* How does a neural network learn ?
* How to implement a neural network ?
* What are some common dimensionality reduction techniques ?
* How to perform feature extraction out of a dataset ?
* What are some evaluation metrics of a ML/DL model (ROC/AUC, precision, recall, etc.) ?

### ANN.py
Definition of artificial neural network class. Implemented in this way to follow basic OOP ideas such as seperation of concerns and reusability of code.

### ANN.ipynb
Implementation of a neural network model. Added markdown comments to increase readability of code. One can observe the training and test process of the network.

### Obesity Dataset
We've selected this dataset to perform our implementation since it has data values that are both numerical and categorical which is a similar case for the medical dataset we're planning to source from NACC.

### decision-tree-on-obesity-dataset.ipynb
It is an implementation of a decision tree on the obesity dataset. It's main purpose is simply to extract most important features of the dataset.

### word-embedding-on-obesity-dataset.ipynb
This is an example application of word embedding on obesity dataset. We replaced different columns with their corresponding word embedding vectors that are sourced from GloVe's library and train the model with replaced columns and reported the new classification statistics such as accuracy and precision.

### Clustering.py
Implementation of K-means clustering algorithm where we visualize the data points on a 2D space and report our findings such as classification report and performance metrics.

### svm.py
Implementation of one-vs-all SVM Classifier on our dataset. Our aim was to visualize the data on a 2D space and see how linearly classifiable the data points are. We generated a plot with decision boundary and reported our findings.

### main.ipynb
The main Jupyter Notebook file where we preprocess the original NACC dataset and train our ANN. We implemented almost whole logic in this file including hyperparameter tuning, reporting training results and thorough preprocessing stages.

*Note :* *Content and descriptions under some titles may change throughout the project. (e.g. folder structure)*
