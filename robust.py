""" robust.py

This is a helper function for solving the robust SLS problem. The code is taken
from the following repo:
    https://github.com/modestyachts/robust-adaptive-lqr
It has been adapted slightly to work with CVX 1.1

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

def robust_constraints(A, B, S, T, gamma, num_hops):
    """ Solve the robust synthesis problem """

    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert len(B.shape) == 2 and B.shape[0] == A.shape[0] and \
            B.shape[0] == B.shape[1]
    assert T >= 1

    n, p = B.shape
    constr = []

    #TODO: we are assuming scalar systems here
    phi_x = cvx.Variable((T, num_hops), name="Phi_x")
    phi_u = cvx.Variable((T, num_hops), name="Phi_u")

    lbd_S, V = np.linalg.eigh(S)
    lbd_A = np.diag(V.T @ A @ V)
    lbd_B = np.diag(V.T @ B @ V)

    # Construct Phi_x and Phi_u from filter weights
    Phi_x = [ np.sum([phi_x[t, i] * V @ np.diag(np.power(lbd_S, i)) @ V.T for i
        in range(num_hops)], axis=0) for t in range(T)]
    Phi_u = [ np.sum([phi_u[t, i] * V @ np.diag(np.power(lbd_S, i)) @ V.T for i
        in range(num_hops)], axis=0) for t in range(T)]

    # hinfnorm constraint (as per Thm 5.8 of Positive trigonometric polynomials
    # and signal processing applications (2007) by B. Dumitrescu.)
    Hbar = cvx.bmat(
        [[(Phi_x[0] - np.eye(n)).T]] +
        [[(Phi_x[k+1] - A @ Phi_x[k] - B @ Phi_u[k]).T] for k in range(T-1)] +
        [[(-A @ Phi_x[T-1] - B @ Phi_u[T-1]).T]])

    Q = cvx.Variable((n*(T+1), n*(T+1)), PSD=True, name="Q")
    # Constraint (5.44)
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

    return constr, Phi_x, Phi_u, phi_x, phi_u, Hbar

def robust_nominal_SLS(Q, R, A, B, S, T, gamma, num_hops):
    constr, Phi_x, Phi_u, phi_x, phi_u, Hbar = \
            robust_constraints(A, B, S, T, gamma, num_hops)
    # H2_cost, TODO: non-identity cost
    htwo_cost = cvx.Variable(name="htwo_cost")
    constr.append( cvx.norm(cvx.bmat([Phi_x + Phi_u]), 'fro') <= htwo_cost )

    # Solve optimization problem
    prob = cvx.Problem(cvx.Minimize(htwo_cost), constr)
    prob.solve(solver=cvx.SCS)

    if prob.status == cvx.OPTIMAL:
        Phi_x = np.vstack([px.value for px in Phi_x])
        Phi_u = np.vstack([pu.value for pu in Phi_u])
        return (True, prob.value, Phi_x, Phi_u, phi_x.value, phi_u.value,
                Hbar.value)
    else:
        #print("could not solve: {}".format(prob.status))
        return (False, None, None, None, None, None, None)

def robust_projection(Px_opt, Pu_opt, A, B, S, T, gamma, num_hops):
    constr, Phi_x, Phi_u, phi_x, phi_u, Hbar = \
            robust_constraints(A, B, S, T, gamma, num_hops)
    # H2_cost, TODO: non-identity cost
    htwo_cost = cvx.Variable(name="htwo_cost")
    constr.append( cvx.norm(cvx.bmat([Phi_x + Phi_u]) - \
            np.vstack([Px_opt, Pu_opt]).T, 'fro') <= htwo_cost )

    # Solve optimization problem
    prob = cvx.Problem(cvx.Minimize(htwo_cost), constr)
    prob.solve(solver=cvx.SCS)

    if prob.status == cvx.OPTIMAL:
        Phi_x = np.vstack([px.value for px in Phi_x])
        Phi_u = np.vstack([pu.value for pu in Phi_u])
        return (True, prob.value, Phi_x, Phi_u, phi_x.value, phi_u.value,
                Hbar.value)
    else:
        #print("could not solve: {}".format(prob.status))
        return (False, None, None, None, None, None, None)

def robust_synth_bisec(Q, R, A, B, S, T, num_hops, cost_lb, cost_ub,
        num_iter=10):
    gamma = cvx.Variable(pos=True)
    htwo_cost, base_constr, Phi_x, Phi_u, phi_x, phi_u = \
            construct_robust_synth(Q, R, A, B, S, T, gamma, num_hops)
    base_constr.append(gamma <= 1-1e-8)

    # Start the bisection at the given upper bound
    best_phi_x, best_phi_u, best_Phi_x, best_Phi_u = None, None, None, None
    best_cost = np.inf
    curr_cost = cost_ub
    for _ in range(num_iter):
        print('Current bounds are ({}, {})'.format(cost_lb, cost_ub))
        print('Solving for {}'.format(curr_cost))

        constr = base_constr + [htwo_cost <= curr_cost * (1-gamma)]
        prob = cvx.Problem(cvx.Minimize(1), constr)

        print('problem is quasi-convex?', prob.is_dqcp())

        prob.solve(solver=cvx.SCS)
        if prob.status == cvx.OPTIMAL:
            print('succeeded')
            best_Phi_x = np.vstack([px.value for px in Phi_x])
            best_Phi_u = np.vstack([pu.value for pu in Phi_u])
            best_phi_x = phi_x.value
            best_phi_u = phi_u.value
            best_cost = curr_cost
            cost_ub = curr_cost
        else:
            print('failed')
            cost_lb = prob.value
        curr_cost = (cost_lb + cost_ub) / 2

    return best_cost, best_Phi_x, best_Phi_u, best_phi_x, best_phi_u
