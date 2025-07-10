from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.linalg import eigh


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        '''
        Initialize the CSP model with the number of components to extract.
        
        Parameters:
        n_components: int, number of components to extract.
        '''
        if not isinstance(n_components, int) or n_components % 2 != 0 or n_components < 2:
            raise ValueError("n_components must be an even integer and greater than 1.")
        self.n_components = n_components
        self.filters_ = None

    def _get_class_covariance(self, X_class, reg_alpha=1e-6):
        """
        Calcula la matriz de covarianza promedio normalizada para una clase de datos.
        X_class: numpy array de forma (n_epochs_class, n_channels, n_samples).
        """
        n_epochs_class, n_channels, n_samples = X_class.shape

        # Calcular la covarianza para cada época y promediar
        cov_matrices = np.array([np.cov(epoch, rowvar=True) for epoch in X_class])
        
        avg_cov_matrix = np.mean(cov_matrices, axis=0)
        # Añade una constante positiva a la diagonal para evitar que la matriz sea singular
        avg_cov_matrix += reg_alpha * np.eye(n_channels)

        # Normalizar por traza (suma de la diagonal)
        avg_cov_matrix /= np.trace(avg_cov_matrix)
        
        return avg_cov_matrix

    def fit(self, x, y):
        '''
        Fit the CSP model to the data. Obtains the projection matrix W
        that maximizes the variance of the projected signals for one class and minimizes it for the other.

        Parameters:
        x: numpy array of shape (n_epochs, n_channels, n_samples).
        y: numpy array of shape (n_epochs,) with the class labels for each epoch.
        
        Returns:
        self: the fitted CSP model with the projection matrix W.
        '''
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"CSP requires exactly two classes. Found {len(unique_classes)}: {unique_classes}")

        # Separar datos por clases y calcular sus covarianzas promedio
        cov_class1 = self._get_class_covariance(x[y == unique_classes[0]], reg_alpha=1e-6)
        cov_class2 = self._get_class_covariance(x[y == unique_classes[1]], reg_alpha=1e-6)
        # La matriz de covarianza captura la distribución espacial de la actividad cerebral y las relaciones entre los diferentes canales.
        # Al separar por ambas clases, obtenemos una representación estadística de la actividad de cada clase para encontrar los filtros que las separen.

        # Calcular la matriz de covarianza total (suma de covarianzas de ambas clases)
        common_covariance = cov_class1 + cov_class2

        # Problema de valores propios generalizado
        # eigh(A, B) resuelve A * w = lambda * B * w
        # Aquí, A = cov_class1 y B = common_covariance (cov_class1 + cov_class2)
        # Los valores propios 'eigenvalues' representan la proporción de varianza de cov_class1 en relación con la varianza total.
        eigenvalues, eigenvectors = eigh(cov_class1, common_covariance)
        
        # Ordenar los valores propios de forma descendente y los vectores propios correspondientes
        # Los filtros con los valores propios MÁS ALTOS son los que maximizan la varianza de la Clase 1.
        # Los filtros con los valores propios MÁS BAJOS son los que maximizan la varianza de la Clase 2.
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Seleccionar los componentes CSP:
        # Tomar n_components // 2 de los primeros (mayor varianza en Clase 1)
        # y n_components // 2 de los últimos (mayor varianza en Clase 2)
        n_selection = self.n_components // 2
        
        # Concatenar los filtros seleccionados
        self.filters_ = np.concatenate((eigenvectors[:, :n_selection], eigenvectors[:, -n_selection:]), axis=1)

        return self

    def transform(self, x):
        '''
        Transform the data using the learned CSP spatial filters.

        Parameters:
        x: numpy array of shape (n_epochs, n_channels, n_samples).
        
        Returns:
        transformed_features: numpy array of shape (n_epochs, n_components) with the
                              log-variance features of the transformed epochs.
        '''
        if self.filters_ is None: # Usar self.filters_ en lugar de self.w_matrix para consistencia
            raise RuntimeError("The CSP model has not been fitted. Call 'fit' first.")

        transformed_features = []
        for epoch in x:
            # epoch: (n_channels, n_samples)
            # self.filters_: (n_channels, n_components)
            # projected_epoch: (n_components, n_samples)
            projected_epoch = self.filters_.T @ epoch
            
            # Calcular la varianza de cada componente proyectado
            variance = np.var(projected_epoch, axis=1)
            
            # Normalizar las varianzas
            sum_variance = np.sum(variance)
            # Evitar división por cero
            if sum_variance == 0:
                normalized_variance = np.zeros_like(variance)
            else:
                normalized_variance = variance / sum_variance

            # Aplicar logaritmo para obtener las características finales
            log_variance = np.log(normalized_variance + 1e-6)

            transformed_features.append(log_variance)
        
        return np.array(transformed_features)