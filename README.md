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

The covariance matrix is defined from the equation below.

$$
Ax=y
$$

The matrix A represents the feature dataset which a ML model will be trained on.
The matrix A can be expressed in terms of the mean, standard deviation, and a matrix $B$ which is the standardized data matrix, where each feature has a mean of 0 and a standard deviation of 1.

$$
A=\sigma\left({B+\mu}\right)\\
\Rightarrow
B=\frac{A-\mu}{\sigma}
$$


$$
A=\frac{B+\mu}{\sigma}
\Rightarrow
Ax=\frac{B+\mu}{\sigma}x=y
$$
