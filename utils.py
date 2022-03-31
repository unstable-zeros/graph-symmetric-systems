import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def H2norm(A, B, C):
    P = solve_discrete_lyapunov(A, B @ B.T)
    return np.sqrt(np.trace(C @ P @ C.T))

def constructSS(H, T, n, m):
    assert H.shape[0] == n * (T+1)
    assert H.shape[1] == m
    Z = np.zeros((n*T, n*T))
    Z[n:, :] = np.eye(n*T)[:n*(T-1), :]
    I = np.vstack([np.eye(n), np.zeros((n*(T-1), m))])
    C = H[n:,:].T
    D = H[:n,:]
    return Z, I, C, D

def computeDelta(A, B, Phi_x, Phi_u, T):
    n = Phi_x.shape[1]
    m = Phi_u.shape[1]
    Delta = np.zeros(((T+1)*n, n))
    Delta[:n,:] = Phi_x[:n,:] - np.eye(n)
    for t in range(T-1):
        Delta[(t+1)*n:(t+2)*n, :] = Phi_x[(t+1)*n:(t+2)*n, :] - \
                A @ Phi_x[t*n:(t+1)*n, :] - B @ Phi_u[t*n:(t+1)*n, :]
    Delta[-n:,:] = -A @ Phi_x[-n:,:] - B @ Phi_u[-n:,:]
    return Delta

def achieved_Phi(A, B, Phi_x, Phi_u, T):
    n = Phi_x.shape[1]
    m = Phi_u.shape[1]
    Delta = computeDelta(A, B, Phi_x, Phi_u, T)
    D0inv = np.linalg.pinv(Delta[:n, :] + + np.eye(n))
    Dhat = Delta[n:, :].T #TODO
    Z = np.zeros((n*T, n*T))
    Z[n:, :] = np.eye(n*T)[:n*(T-1), :]
    I = np.vstack([np.eye(n), np.zeros((n*(T-1), m))])
    ID0inv = I @ D0inv
    Phi_x_hat = Phi_x.T #TODO
    Phi_u_hat = Phi_u.T #TODO
    A_x = np.block([[Z, ID0inv @ Dhat], [np.zeros((n*T, n*T)), Z - ID0inv @ Dhat]])
    B_x = np.vstack([ID0inv, -ID0inv])
    C_x = np.block([Phi_x_hat, np.zeros((n, n*T))])
    C_u = np.block([Phi_u_hat, np.zeros((n, n*T))])
    D_x = np.zeros((n,n))
    return A_x, B_x, C_x, C_u, D_x

def compute_cost(A, B, Phi_x, Phi_u, T):
    A_x, B_x, C_x, C_u, D_x = achieved_Phi(A, B, Phi_x, Phi_u, T)
    H2x = H2norm(A_x, B_x, C_x)
    H2u = H2norm(A_x, B_x, C_u)
    return np.sqrt(H2x**2 + H2u**2)
