import numpy as np
import networkx as nx

def _sample_adjacency_matrix(N, degree):
    """ Samples a random adjacency matrix; See below for details """
    rand_vec = np.random.random(N)
    Adj = np.zeros((N,N))
    for i in range(N):
      inds = np.abs(rand_vec-rand_vec[i]).argpartition(degree)[:degree]
      Adj[np.ones(degree, dtype=int)*i, inds] = 1
      Adj[inds, np.ones(degree, dtype=int)*i] = 1
    return Adj

def _generate_env(N, degree, GSO='lap'):
    """ Generates an environment, i.e., A, B, Q, R, S
    Parameters:
        - N:        integer, number of agents
        - degree:   desired degree of the communication graph
    Returns:
        - G:        nx.Graph, the graph
        - S:        np.array, the adjacency matrix of G
    """
    Adj = _sample_adjacency_matrix(N, degree)
    G = nx.Graph(Adj)
    while(not nx.is_connected(G)):
        Adj = _sample_adjacency_matrix(N, degree)
        G = nx.Graph(Adj)
    assert np.all(Adj == Adj.T)

    # Compute the normalized Laplacian
    degrees = Adj @ np.ones(N) - 1  # Subtract the self-loop
    L = np.diag(degrees) - Adj + np.eye(N)
    D_sqrt_inv = np.diag(1 / np.sqrt(degrees))

    if GSO == 'lap':
        S = D_sqrt_inv @ L @ D_sqrt_inv
    elif GSO == 'adj':
        S = D_sqrt_inv @ Adj @ D_sqrt_inv
    else:
        pattern = (Adj == 0)
        conds = []
        Ss = []
        # Generate 5 random GSO's and pick the one that leads to the lowest
        # condition number for the Vandermonde matrix
        for _ in range(100):
            S = np.random.normal(0, 1, (N,N))
            S = S + S.T
            S[pattern] = 0
            S = S / np.max(np.abs(np.linalg.eigh(S)[0]))
            lbd_S, _ = np.linalg.eigh(S)
            A = np.vstack([np.power(lbd_S, i) for i in range(N)]).T
            conds.append(np.linalg.cond(A))
            Ss.append(S)
        S = Ss[np.argmin(conds)]

    ## Perturb the GSO
    #var = 1e-1
    #S = S + np.random.normal(0, var, S.shape)
    # Compute the normalized adjacency matrix

    #Or, use random S
    #S = np.random.normal(0, 1, (N,N))
    #S = S + S.T
    #S = S / np.linalg.norm(S, 2)
    #U = np.linalg.eigh(S)[1]
    #S = U @ np.diag(np.arange(N)) @ U.T

    # Generate the A, B matrices to be commute with S
    eig_vecs = np.linalg.eigh(S)[1]
    A = eig_vecs @ np.diag(np.random.randn(N)) @ eig_vecs.T
    B = eig_vecs @ np.diag(np.random.randn(N)) @ eig_vecs.T
    # Take Q and R to be identity for now
    Q = np.eye(N)
    R = np.eye(N)
    return G, S, A, B, Q, R

def Phi_achievable(A, B, Phi_x, Phi_u):
    #TODO
    n = A.shape[0]
    T = Phi_x.shape[0] / n
    achievable = True

def get_traj(Phi_x, Phi_u, x0):
    xtraj = Phi_x @ x0
    utraj = Phi_u @ x0
    return xtraj, utraj

def get_traj_rob(A, B, Phi_x, Phi_u, x0, T_sim):
    # Compute the dimensions
    nT, n = Phi_x.shape
    _, m = Phi_u.shape
    T = int(nT / n)

    # Construct the controller internal dynamics matrices
    R1 = Phi_x[:n, :]
    R1_inv = np.linalg.inv(R1)
    I_tilde = np.vstack([ np.eye(n), np.zeros((n*(T-2), n)) ])
    R_hat = (Phi_x[n:, :] @ R1_inv).T
    M_hat = (Phi_u[n:, :] @ R1_inv).T
    M1 = Phi_u[:n, :] @ R1_inv
    Z = np.zeros((n*(T-1), n*(T-1)))
    Z[n:, :] = np.eye(n*(T-1))[:n*(T-2), :]
    A_u = Z - I_tilde @ R_hat
    B_u = -I_tilde
    C_u = M1 @ R_hat - M_hat
    D_u = M1

    # Initialize the states
    x_traj = np.zeros((T_sim, n))
    u_traj = np.zeros((T_sim, m))
    curr_x = x0
    curr_xk = np.zeros(n*(T-1))
    for t in range(T_sim):
        x_traj[t, :] = curr_x
        curr_u = C_u @ curr_xk + D_u @ curr_x
        u_traj[t, :] = curr_u
        curr_xk = A_u @ curr_xk + B_u @ curr_x
        curr_x = A @ curr_x + B @ curr_u
    return x_traj, u_traj
