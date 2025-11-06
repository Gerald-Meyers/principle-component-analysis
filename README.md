# A basic introduction

This Repo is a demonstration of the PCA algorithm to clean data, find the eigenvectors of a dataset which can be used to optimize the learning rate of a neural network. This is a project meant to 

SciKit Learn does have a default method implemented for PCA.
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

The notable use of this project is to perform the PCA and save the PCA data to a file for later use.

## The Covariance Matrix

https://en.wikipedia.org/wiki/Covariance_matrix

The matrix A represents the feature dataset which a ML model will be trained on.
The matrix A can be expressed in terms of the mean, standard deviation, and a matrix $B$ which is the standardized data matrix, where each feature has a mean of 0 and a standard deviation of 1.

$$
B=\frac{A-\mu}{\sigma}
$$

Where $\mu$ is the vector of feature means and $\sigma$ is the vector of feature standard deviations.

The covariance matrix, $C$, measures the variance and covariance between the features of the dataset. For PCA, we compute the covariance matrix of the standardized data, $B$.

The formula for the covariance matrix is:

$$
C = \frac{1}{n-1} B^T B
$$

Where $n$ is the number of samples (rows) in the dataset. The term $B^T B$ is often called the "scatter matrix". In the context of PCA, we are interested in the eigenvectors of this matrix. The scaling factor $\frac{1}{n-1}$ scales the eigenvalues but does not change the direction of the eigenvectors, so it is sometimes omitted for simplicity, as seen in the `_covariance_matrix` method in the Python code.

The eigenvectors of the covariance matrix are the principal components of the dataset.

---

The equation $Ax=y$ can represent a linear regression model, where we want to find a vector of coefficients $x$ that maps our features $A$ to a target variable $y$. While both use the data matrix $A$, the calculation of the covariance matrix for PCA is a separate process that analyzes the internal structure of $A$.
