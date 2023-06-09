import torch
import numpy as np

def pca(X: torch.Tensor, k: int) -> torch.Tensor:
    #   X: samples to transform
    #   k: dimension of each output sample
    #   output: samples after transformation

    X = X.view(X.size(0), -1)

    # Compute the covariance matrix
    X_centered = X - X.mean(dim=0)
    cov_matrix = torch.matmul(X_centered.t(), X_centered) / (X.shape[0] - 1)

    # Compute the eigenvectors and eigenvalues and sort them
    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues.real, descending=True)
    eigenvalues_sorted = eigenvalues.real[sorted_indices]
    eigenvectors_sorted = eigenvectors.real[:, sorted_indices]

    # Select the top k eigenvectors
    components = eigenvectors_sorted[:, :k]
    X_pca = torch.matmul(X_centered, components)

    return X_pca

def sne():
    pass