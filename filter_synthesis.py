""" filter_synthesis.py

This is a helper function synthesizing controllers for simultaneously
diagonalizable systems under an FIR truncation approximation.

"""

import numpy as np
import cvxpy as cvx
import scipy.linalg

def psd_sqrt(P):
    assert len(P.shape) == 2
    assert P.shape[0] == P.shape[1]
    w, v = np.linalg.eigh(P)
    assert (w >= 0).all()
    return v.dot(np.diag(np.sqrt(w))).dot(v.T)

def direct_param(Q, R, A, B, S, T):
    """
    Solves the SLS synthesis problem for length T FIR filters
    using CVXPY

    """

    assert len(Q.shape) == 2 and Q.shape[0] == Q.shape[1]
    assert len(R.shape) == 2 and R.shape[0] == R.shape[1]
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert len(B.shape) == 2 and B.shape[0] == A.shape[0] and \
            B.shape[0] == B.shape[1]
    assert Q.shape[0] == A.shape[0]
    assert R.shape[0] == B.shape[1]
    assert T >= 1

    n, p = B.shape

    Q_sqrt = psd_sqrt(Q)
    R_sqrt = psd_sqrt(R)

    #TODO: we are assuming scalar systems here
    # Phi_x = \sum_{k=1}^{T} Phi_x[k] z^{-k}
    phi_x = cvx.Variable((T, n), name="Phi_x")

    # Phi_u = \sum_{k=1}^{T} Phi_u[k] z^{-k}
    phi_u = cvx.Variable((T, n), name="Phi_u")

    lbd_S, V = np.linalg.eigh(S)
    lbd_A = np.diag(V.T @ A @ V)
    lbd_B = np.diag(V.T @ B @ V)

    # htwo_cost
    htwo_cost = cvx.Variable(name="htwo_cost")

    spec_x = [cvx.sum([phi_x[t, i] * np.power(lbd_S, i) for i in range(n)],
        axis=0) for t in range(T)]
    spec_u = [cvx.sum([phi_u[t, i] * np.power(lbd_S, i) for i in range(n)],
        axis=0) for t in range(T)]

    # subspace constraint:
    constr = []
    constr.append(spec_x[0] == np.ones(n))
    for k in range(T-1):
        constr.append(
            spec_x[k+1] == cvx.multiply(lbd_A, spec_x[k]) +
            cvx.multiply(lbd_B, spec_u[k]))

    constr.append(
        cvx.multiply(lbd_A, spec_x[T-1]) + cvx.multiply(lbd_B, spec_u[T-1]) == 0)

    # htwo constraint:
    # TODO: fix this for non-identity Q, R
    beta = 1e-5
    constr.append(
        cvx.square(cvx.norm(cvx.bmat([spec_x] + [spec_u]), 'fro')) <= htwo_cost)

    prob = cvx.Problem(cvx.Minimize(htwo_cost), constr)

    eps=1e-4
    mosek_params = {
            "MSK_DPAR_INTPNT_TOL_DFEAS": eps,
            "MSK_DPAR_INTPNT_TOL_PFEAS": eps,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": eps,
    }
    prob.solve(solver=cvx.MOSEK, verbose=False, mosek_params = mosek_params)
    #prob.solve(solver=cvx.SCS, verbose=False, eps=eps)

    if prob.status == cvx.OPTIMAL:
        #print("successfully solved!")
        phi_x = phi_x.value
        phi_u = phi_u.value

        # Construct Phi_x and Phi_u
        Phi_x = np.vstack([V @ np.diag(spec_x[t].value) @ V.T for t in range(T)])
        Phi_u = np.vstack([V @ np.diag(spec_u[t].value) @ V.T for t in range(T)])

        #######################
        print(htwo_cost.value)
        #######################

        return (True, prob.value, Phi_x, Phi_u, phi_x, phi_u)
    elif prob.status == cvx.OPTIMAL_INACCURATE:
        print("optimal_inaccurate...")
        phi_x = phi_x.value
        phi_u = phi_u.value

        # Construct Phi_x and Phi_u
        Phi_x = np.vstack([V @ np.diag(spec_x[t].value) @ V.T for t in range(T)])
        Phi_u = np.vstack([V @ np.diag(spec_u[t].value) @ V.T for t in range(T)])

        #######################
        print(htwo_cost.value)
        #######################

        return (True, prob.value, Phi_x, Phi_u, phi_x, phi_u)

    else:
        print("could not solve: {}".format(prob.status))
        return (False, None, None, None, None, None)

def gf_synth(Q, R, A, B, S, T):
    """
    Solves the SLS synthesis problem for length T FIR filters
    using CVXPY

    """

    assert len(Q.shape) == 2 and Q.shape[0] == Q.shape[1]
    assert len(R.shape) == 2 and R.shape[0] == R.shape[1]
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert len(B.shape) == 2 and B.shape[0] == A.shape[0] and \
            B.shape[0] == B.shape[1]
    assert Q.shape[0] == A.shape[0]
    assert R.shape[0] == B.shape[1]
    assert T >= 1

    n, p = B.shape

    Q_sqrt = psd_sqrt(Q)
    R_sqrt = psd_sqrt(R)

    #TODO: we are assuming scalar systems here
    # Phi_x = \sum_{k=1}^{T} Phi_x[k] z^{-k}
    phi_x = cvx.Variable((T, n), name="Phi_x")

    # Phi_u = \sum_{k=1}^{T} Phi_u[k] z^{-k}
    phi_u = cvx.Variable((T, n), name="Phi_u")

    #####################################
    h_x = cvx.Variable((T, n), name="hx")
    h_u = cvx.Variable((T, n), name="hu")
    ####################################

    V = np.linalg.eigh(S)[1]
    lbd_A = np.diag(V.T @ A @ V)
    lbd_B = np.diag(V.T @ B @ V)

    # htwo_cost
    htwo_cost = cvx.Variable(name="htwo_cost")

    # subspace constraint:
    constr = []
    constr.append(phi_x[0, :] == np.ones(n))
    for k in range(T-1):
        constr.append(
            phi_x[k+1, :] == cvx.multiply(lbd_A, phi_x[k, :]) +
            cvx.multiply(lbd_B, phi_u[k, :]))
    constr.append(
        cvx.multiply(lbd_A, phi_x[T-1,:]) + cvx.multiply(lbd_B, phi_u[T-1,:]) == 0)

    #########################################
    lbd_S, _ = np.linalg.eigh(S)
    constr += [cvx.sum([h_x[t, i] * np.power(lbd_S, i) for i in range(n)],
        axis=0) == phi_x[t] for t in range(T)]
    constr += [cvx.sum([h_u[t, i] * np.power(lbd_S, i) for i in range(n)],
        axis=0) == phi_u[t] for t in range(T)]
    #########################################

    # htwo constraint:
    # TODO: fix this for non-identity Q, R
    constr.append(
        cvx.norm(cvx.bmat([[phi_x], [phi_u]]), 'fro') <= htwo_cost)

    prob = cvx.Problem(cvx.Minimize(htwo_cost), constr)
    prob.solve(solver=cvx.MOSEK)

    if prob.status == cvx.OPTIMAL:
        #print("successfully solved!")
        phi_x = phi_x.value
        phi_u = phi_u.value

        # Construct Phi_x and Phi_u
        Phi_x = np.vstack([V @ np.diag(phi_x[t,:]) @ V.T for t in range(T)])
        Phi_u = np.vstack([V @ np.diag(phi_u[t,:]) @ V.T for t in range(T)])

        return (True, prob.value, Phi_x, Phi_u, phi_x, phi_u)
    else:
        print("could not solve: {}".format(prob.status))
        return (False, None, None, None, None, None)

def solve_ft_weights(Phi, S, trunc=None):
    n = S.shape[0]
    T = int(Phi.shape[0] / n)
    lbd_S, V = np.linalg.eigh(S)
    if trunc is None:
        trunc = n
    assert trunc <= n

    A = np.vstack([np.power(lbd_S, i) for i in range(trunc)]).T

    # Solve for the filter tap weights
    h = np.zeros((T, trunc))
    Phi_recon = np.zeros(Phi.shape)
    for t in range(T):
        phi = np.diag(V.T @ Phi[t*n:(t+1)*n] @ V)
        h[t] = np.linalg.lstsq(A, phi, rcond=-1)[0]
        Phi_recon[t*n:(t+1)*n] = V @ np.diag(A @ h[t]) @ V.T

    return h, Phi_recon

def hinfnorm_Delta(Phi_x, Phi_u, n, T):
    assert Phi_x.shape[0] == n*T
    assert Phi_x.shape[1] == n

    Delta = np.vstack( [(Phi_x[:n] - np.eye(n)).T] +
            [(Phi_x[(k+1)*n:(k+2)*n] - A @ Phi_x[k*n:(k+1)*n] - B @
                Phi_u[k*n:(k+1)*n]).T for k in range(T-1)] +
            [(-A@Phi_x[-n:] - B @ Phi_u[-n:]).T] )

    Q = cvx.Variable((n*(T+1), n*(T+1)), PSD=True, name="Q")

    # Case k==0: the block diag of Q has to sum to gamma^2 * eye(n)
    gamma_sq = gamma ** 2
    constr.append(
        sum([Q[n*t:n*(t+1), n*t:n*(t+1)] for t in range(T+1)]) == gamma_sq*np.eye(n))

    # Case k>0: the block off-diag of Q has to sum to zero
    for k in range(1, T+1):
        constr.append(
            sum([Q[n*t:n*(t+1), n*(t+k):n*(t+1+k)] for t in range(T+1-k)]) == np.zeros((n, n)))

    # Constraint (5.45)
    constr.append(
        cvx.bmat([
            [Q, Hbar],
            [Hbar.T, np.eye(n)]]) == cvx.Variable( (n*(T+1) + (n), n*(T+1)
                + (n)), PSD=True))

    return htwo_cost, constr, Phi_x, Phi_u, phi_x, phi_u, Hbar
