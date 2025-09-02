# OptimalTransport.py

import numpy as np
from scipy.optimize import linear_sum_assignment

def constructOMEGA(ns, nt):
    """Constructs the cost matrix for optimal transport."""
    OMEGA = np.zeros((ns, nt))
    for i in range(ns):
        for j in range(nt):
            OMEGA[i, j] = np.linalg.norm(i - j)  # Example cost based on distance
    return OMEGA

def solveOT(n_s, n_t, S_, h_, X):
    """Solves the optimal transport problem."""
    C = np.zeros((n_s, n_t))
    for i in range(n_s):
        for j in range(n_t):
            C[i, j] = np.sum((X[i] - X[n_s + j]) ** 2)  # Cost based on squared distance

    row_ind, col_ind = linear_sum_assignment(C)
    GAMMA = np.zeros((n_s, n_t))
    for r, c in zip(row_ind, col_ind):
        GAMMA[r, c] = h_[r] * h_[n_s + c]  # Assigning transport plan based on optimal assignment

    return {'GAMMA': GAMMA, 'B': col_ind}  # Return transport plan and assignments