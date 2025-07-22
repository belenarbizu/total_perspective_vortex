from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.linalg import eigh

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, x, y=None):
        #Standardize the data
        self.mean = np.mean(x, axis=0)
        X_standardize = x - self.mean

        #Compute the covariance matrix
        cov_matrix = np.cov(X_standardize, rowvar=False)

        #Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(cov_matrix)

        #Sort the eigenvalues and eigenvectors
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]

        #Select the n_components
        self.components = eigenvectors[:, :self.n_components]
        return self

    def transform(self, x):
        #Project the data onto the selected components
        x_centered = x - self.mean
        return np.dot(x_centered, self.components)