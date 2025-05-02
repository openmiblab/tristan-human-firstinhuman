
"""
Linear Discriminant Analysis (LDA) for Paired Samples

Concept

LDA is typically used to find a linear combination of variables that best 
separates two or more groups. For paired samples, the key idea is to:

Compute the differences between paired observations.
Find a linear combination of the variables that maximizes the separation 
between conditions while considering the covariance structure.
This approach is closely related to Hotelling’s T2T test, but instead of just 
testing for significance, it provides an explicit discriminant function 
(a weighted sum of variables) that gives the largest possible difference 
between conditions.

"""
import math
import numpy as np
from scipy.linalg import inv

def lda_paired(X1, X2):
    """
    Perform LDA for paired samples.
    
    Parameters:
    X1: np.array of shape (n_samples, n_features) - Measurements in condition 1
    X2: np.array of shape (n_samples, n_features) - Measurements in condition 2
    
    Returns:
    w: np.array of shape (n_features,) - Linear discriminant weights
    y: np.array of shape (n_samples,) - Discriminant scores for each subject
    """
    D = X2 - X1  # Compute differences
    mean_D = np.mean(D, axis=0)  # Mean difference vector
    S_D = np.cov(D, rowvar=False)  # Covariance matrix of differences

    # Compute LDA weight vector
    w = inv(S_D).dot(mean_D)

    # Project data onto discriminant axis
    y = D.dot(w)

    return w, y

# # Example usage:
# np.random.seed(42)
# X1 = np.random.randn(20, 3)  # 20 subjects, 3 variables (pre-test)
# X2 = X1 + np.random.randn(20, 3) * 0.5  # Post-test with some change

# w, y = lda_paired(X1, X2)
# print("LDA Weights:", w)
# print("Discriminant Scores:", y)

def first_digit(x):
    if np.isnan(x):
        return x
    return -int(math.floor(math.log10(abs(x))))

def round_sig(x, n):
    # Round to n significant digits
    if x==0:
        return x
    if np.isnan(x):
        return x
    return round(x, first_digit(x) + (n-1))

def round_to_first_digit(x):
    if np.isnan(x):
        return x
    # Round to first significant digit
    n = first_digit(x)
    y = round(x, n)
    if n<=0:
        return int(y)
    return y

def round_meas(x, xerr):
    if np.isnan(x):
        return x
    # Round measurement with known error
    if xerr==0:
        return x, xerr
    n = first_digit(xerr)
    y = round(x, n)
    yerr = round(xerr, n)
    if n<=0:
        return int(y), int(yerr)
    else:
        return y, yerr
    
def around_sig(x, n):
    return np.array([round_sig(v,n) for v in x])

def around_meas(x, xerr):
    n = len(x)
    if len(xerr) != n:
        raise ValueError(
            "The array with error values must have the same length as the "
            "array with measurements."
        )
    y = np.empty(n)
    yerr = np.empty(n)
    for i in range(n):
        y[i], yerr[i] = round_meas(x[i], xerr[i])
    return y, yerr
