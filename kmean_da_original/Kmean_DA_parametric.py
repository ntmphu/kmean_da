# This file contains functions for generating data, computing intervals, and performing parametric analysis related to K-means clustering and domain adaptation.

import numpy as np
import OptimalTransport
import util
import intersection
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import scipy.stats as stats
from multiprocessing import Pool
from tqdm import tqdm 
import sys
from mpmath import mp
import Kmean_DA_oc
import Kmean_clustering

def generate(n, m, p):
    """return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t"""

    Xs = np.random.normal(loc = 0, scale = 1, size = (n, p))
    Xt = np.random.normal(loc = 0, scale = 1, size = (m, p))

    Sigma_s = np.identity(n)
    Sigma_t = np.identity(m)

    return Xs, Xt, Sigma_s, Sigma_t

def compute_z_interval(n, K, a, b, initial_centroids, labels_all, members_all):
    trunc_interval = [(-np.inf, np.inf)]
   
    for i in range(n):
        if  i == initial_centroids[labels_all[0][i]]:
            continue
        e_i_ci0 = np.zeros((1,n))
        e_i_ci0[0][i] = 1
        
        e_i_ci0[0][initial_centroids[labels_all[0][i]]] = -1
        A1 = np.dot(e_i_ci0.T, e_i_ci0)
        p1, q1, o1 = util.construct_p_q_t(a, b, A1)
        for k in range(K):
            if k == labels_all[0][i]:
                continue
            e_i_ck0 = np.zeros((1,n))
            e_i_ck0[0][i] = 1
            e_i_ck0[0][initial_centroids[k]] = -1
            A2 = np.dot(e_i_ck0.T, e_i_ck0)
            
            p2, q2, o2 = util.construct_p_q_t(a, b, A2)
            
            p = p1 - p2
            q = q1 - q2
            o = o1 - o2
            
            res = util.solve_quadratic_inequality(p[0][0], q[0][0], o[0][0])
            if res == "No solution":
                print(p, q, t)
            else:
                trunc_interval = util.interval_intersection(trunc_interval,res)
                
    
    for t in range(1, len(labels_all)):
        for i in range(n):
            e_i = np.zeros((1,n))
            e_i[0][i] = 1
            
            gamma_i = np.zeros((1,n))
            label_i = labels_all[t][i] 
            
            C_i_t_minus = list(members_all[t-1][label_i])  
            if len(C_i_t_minus) == 0:
                    continue
    
            gamma_i[:,C_i_t_minus] = 1
        
            E_temp_1 = e_i - gamma_i/len(C_i_t_minus)
            
            E1 = np.dot(E_temp_1.T, E_temp_1)
            p3, q3, o3 = util.construct_p_q_t(a, b, E1)
            for k in range(K):
                e_i = np.zeros((1,n))
                e_i[0][i] = 1
                
                gamma_k = np.zeros((1,n))         
                C_k_t_minus = list(members_all[t-1][k])
                if len(C_k_t_minus) == 0:
                    continue
    
                if k == label_i:
                    continue
                gamma_k[:,C_k_t_minus] = 1
                
                E_temp_2 = e_i - gamma_k/len(C_k_t_minus)
                
                E2 = np.dot(E_temp_2.T, E_temp_2)
                
                p4, q4, o4 = util.construct_p_q_t(a, b, E2)
                
                p_comma = p3 - p4
                q_comma = q3 - q4
                o_comma = o3 - o4

                res = util.solve_quadratic_inequality(p_comma[0][0], q_comma[0][0], o_comma[0][0])
                if res == "No solution":
                    print(p_comma, q_comma, o_comma)
                else:
                    trunc_interval = util.interval_intersection(trunc_interval,res)
    return trunc_interval

def interval_DA(ns, nt, X_, B, S_, h_, a, b):
    Bc = np.delete(np.array(range(ns*nt)), B)

    OMEGA = OptimalTransport.constructOMEGA(ns,nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X_.shape[1]-1):
        c_ += (OMEGA.dot(X_[:, [i]])) * (OMEGA.dot(X_[:, [i]]))

    Omega_a = OMEGA.dot(a)
    Omega_b = OMEGA.dot(b)

    w_tilde = c_ + Omega_a * Omega_a
    r_tilde = Omega_a * Omega_b + Omega_b * Omega_a
    o_tilde = Omega_b * Omega_b
    S_B_invS_Bc = np.linalg.inv(S_[:, B]).dot(S_[:, Bc])

    w = (w_tilde[Bc, :].T - w_tilde[B, :].T.dot(S_B_invS_Bc)).T
    r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(S_B_invS_Bc)).T
    o = (o_tilde[Bc, :].T - o_tilde[B, :].T.dot(S_B_invS_Bc)).T
    list_intervals = []

    interval = [(-np.inf, np.inf)]
    for i in range(w.shape[0]):
        g3 = - o[i][0]
        g2 = - r[i][0]
        g1 = - w[i][0]
        itv = intersection.solve_quadratic_inequality(g3,g2,g1)

        interval = intersection.interval_intersection(interval, itv)
    return interval

def parametric(K, n_s, n_t, a, b, S_, h_, u, v, cluster_u_obs, cluster_v_obs, z_min, z_max):
    TD = []
    detectedinter = []
    z =  z_min
    zmax = z_max
    countitv = 0
    while z < zmax:
        z += 0.0001

        for i in range(len(detectedinter)):
            if detectedinter[i][0] <= z <= detectedinter[i][1]:
                z = detectedinter[i][1] + 0.0001
                detectedinter = detectedinter[i:]
                break
        if z > zmax:
            break
        X_z = a + b*z
        
        GAMMA_u, B_u = OptimalTransport.solveOT(n_s, n_t, S_, h_, X_z).values()

        Xtildeinloop = np.dot(GAMMA_u, X_z)

        initial_centroids_v, labels_all_v, members_all_v, cluster_u_v, cluster_v_v = Kmean_clustering.Kmean_selected_uv(Xtildeinloop, K, u, v, n_s, n_t)
        intervalinloop, itvda, itv_Kmean = Kmean_DA_oc.overconditioning(K, n_s, n_t, a, b, X_z, B_u, S_, h_,  GAMMA_u, initial_centroids_v, labels_all_v, members_all_v)
        countitv +=1
        detectedinter = intersection.interval_union(detectedinter, intervalinloop)

        if not(np.array_equal(np.sort(cluster_u_obs), np.sort(cluster_u_v)) and np.array_equal(np.sort(cluster_v_obs), np.sort(cluster_v_v))):
            continue

        TD = intersection.interval_union(TD, intervalinloop)
    
    return TD

def run(n_s, n_t, p, K):
    Xs, Xt, Sigma_s, Sigma_t = generate(n_s, n_t, p)
    X = np.concatenate((Xs, Xt), axis=0)
    Sigma = block_diag(Sigma_s, Sigma_t)
    Xs_vec = Xs.flatten().copy().reshape((n_s * p, 1))
    Xt_vec = Xt.flatten().copy().reshape((n_t * p, 1))
    X_vec = np.vstack((Xs_vec, Xt_vec)).copy()

    C = np.zeros((n_s, n_t))

    for i in range(n_s):
        e_x = Xs[i, :]
        for j in range(n_t):
            e_y = Xt[j, :]
            C[i, j] = np.sum((e_x - e_y)**2)

    h = np.concatenate((np.ones((n_s, 1)) / n_s, np.ones((n_t, 1)) / n_t), axis=0)
    S = OptimalTransport.convert(n_s, n_t)

    S_ = S[:-1].copy()
    h_ = h[:-1].copy()

    GAMMA, B = OptimalTransport.solveOT(n_s, n_t, S_, h_, X).values()

    X_tilde = np.dot(GAMMA, X)

    n = n_s + n_t
    u, v = np.random.choice(K, 2, replace=False)
    
    initial_centroids, labels_all, members_all, cluster_u_obs, cluster_v_obs = Kmean_clustering.Kmean_selected_uv(X_tilde, K, u, v, n_s, n_t)
    if len(cluster_u_obs) == 0 or len(cluster_v_obs) == 0:
        return None
    eta_c_u = np.zeros((n_t, 1))
    eta_c_u[cluster_u_obs] = 1
    eta_c_v = np.zeros((n_t, 1))
    eta_c_v[cluster_v_obs] = 1
    eta_uv = eta_c_v/len(cluster_v_obs) -  eta_c_u/len(cluster_u_obs)

    eta = np.vstack((np.zeros((n_s, 1)), eta_uv))

    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta).item()
    
    I_nplusm = np.identity(n)
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), X)

    etaTX = np.dot(eta.T, X).item()
    
    h = np.concatenate((np.ones((n_s, 1)) / n_s, np.ones((n_t, 1)) / n_t), axis=0)
    S = OptimalTransport.convert(n_s, n_t)

    z_min = -20
    z_max = 20
    finalinterval = parametric(
                K, n_s, n_t, a, b, S_, h_, u, v, cluster_u_obs, cluster_v_obs, z_min, z_max
            )
    selective_p_value = util.compute_p_value(finalinterval, etaTX, etaT_Sigma_eta)
    if selective_p_value is None:
        print("interval error")
    return selective_p_value

def run_wrapper(args):
    n_s, n_t, p, K = args
    return run(n_s, n_t, p, K)

if __name__ == "__main__":
    max_iteration = 300
    Alpha = 0.05
    count = 0
    n_s = 30
    n_t = 20
    p = 1
    K = 3
    list_p_value = []
    underalpha = 0
    args = [(n_s, n_t, p, K) for _ in range(max_iteration)]
    
    with Pool(initializer=np.random.seed) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
            results.append(result)
    
    list_p_value = [p_value for p_value in results if p_value is not None]

    count = sum(1 for p_value in list_p_value if p_value <= Alpha)
    
    print('\nFalse positive rate:', count/len(list_p_value), len(list_p_value))

    print(stats.kstest(list_p_value, stats.uniform(loc=0.0, scale=1.0).cdf))

    plt.hist(list_p_value, bins=20)

    plt.show()