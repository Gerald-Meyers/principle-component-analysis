from numpy import array, mean, std


class PCA:

    def __init__(self, dataset, n_components):
        self.dataset = dataset
        self.n_components = n_components
        self.component_vectors = array([])
        self.explained_variance = array([])
        self.transformed_dataset = array([])

    def fit(self):
        ...

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
        return normalized_dataset.T @ normalized_dataset return normalized_dataset.T @ normalized_dataset return normalized_dataset.T @ normalized_dataset
