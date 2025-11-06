from typing import Any, Callable, Iterable, Optional, Union

from numpy import (  # float80,; float96,; float256,
    array,
    asarray,
    average,
    float16,
    float32,
    float64,
    float128,
    int_,
    mean,
    ndarray,
    random,
    std,
)
from numpy.linalg import eig
from numpy.typing import NDArray

PythonScalars = (int, float)
PythonScalar: type = Union[*PythonScalars]
PythonArray = Iterable[PythonScalar]

FloatScalars = (
    int_,
    float16,
    float32,
    float64,
    # float80,
    # float96,
    # float128,
    # float256
)
FloatScalar = Union[*FloatScalars]
FloatArray = NDArray[FloatScalar]
CallableScalar = Callable[[FloatArray], FloatScalar]


class PCA:

    def __init__(self, dataset, n_components):
        self.dataset = dataset
        self.n_samples, self.n_features = dataset.shape
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
        # 1. Normalize the original dataset
        normalized_dataset = self._normalize()

        # 2. Get the top n_components
        top_n_components = self.component_vectors[0: self.n_components]

        # 3. Project the data onto the principal components
        self.transformed_dataset = normalized_dataset @ top_n_components.T
        return self.transformed_dataset

    def fit_transform(self):
        self.fit()
        self.transform()
        return self.get_transformed_dataset()

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

        self.normalized_dataset = (self.dataset - mean_vector) / std_vector

    def _covariance_matrix(self):
        self.covariance_matrix = \
            self.normalized_dataset.T @ self.normalized_dataset / \
            (self.n_samples - 1)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self.dataset))
            f.write(str(self.component_vectors))
            f.write(str(self.explained_variance))
            f.write(str(self.transformed_dataset))
            f.write(str(self.n_components))
