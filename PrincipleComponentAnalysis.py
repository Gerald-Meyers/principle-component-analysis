from typing import Any, Callable, Iterable, Optional, Union

from numpy import (array, asarray, average,  # float80,; float96,; float256,
                   float16, float32, float64, float128, int_, mean, ndarray,
                   random, std)
from numpy.linalg import eig
from numpy.typing import NDArray

PythonScalars = (int, float)
PythonScalar: type = Union[*PythonScalars]
PythonArray = Iterable[PythonScalar]


class PCA:

    def __init__(self, dataset, n_components):
        self.dataset = dataset
        self.n_components = n_components
        self.component_vectors = array([])
        self.explained_variance = array([])
        self.transformed_dataset = array([])

    def fit(self):
        # 1. Calculate covariance matrix
        cov_matrix = self._covariance_matrix()

        # 2. Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(cov_matrix)

        # 3. Sort eigenvectors by descending eigenvalues
        # Transpose eigenvectors so they are rows
        eigenvectors = eigenvectors.T
        idxs = array(eigenvalues).argsort()[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 4. Store the principal components and explained variance
        self.component_vectors = eigenvectors

        total_variance = sum(eigenvalues)
        self.explained_variance = eigenvalues / total_variance

    def transform(self):
        ...

    def fit_transform(self):
        ...
        return self.transformed_dataset

    def get_component_vectors(self):
        return self.component_vectors

    def get_explained_variance(self):
        return self.explained_variance

    def get_transformed_dataset(self):
        return self.transformed_dataset

    def get_dataset(self):
        return self.dataset

    def get_n_components(self):
        return self.n_components

    def _normalize(self):
        mean_vector = mean(self.dataset, axis=0)
        std_vector = std(self.dataset, axis=0)

        return (self.dataset - mean_vector) / std_vector

    def _covariance_matrix(self):
        normalized_dataset = self._normalize()
        n_samples = normalized_dataset.shape[0]
        return normalized_dataset.T @ normalized_dataset / (n_samples - 1)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(dataset)
            f.write(component_vectors)
            f.write(explained_variance)
            f.write(transformed_dataset)
            f.write(n_components)
