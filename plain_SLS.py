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

def sls_synth(Q, R, A, B, S, T):
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
    Phi_x = cvx.Variable((T*n, n), name="Phi_x")

    # Phi_u = \sum_{k=1}^{T} Phi_u[k] z^{-k}
    Phi_u = cvx.Variable((T*n, n), name="Phi_u")

    # htwo_cost
    htwo_cost = cvx.Variable(name="htwo_cost")

    # subspace constraint:
    # [zI - Ah, -Bh] * [Phi_x; Phi_u] = I
    constr = []
    constr.append(Phi_x[:n, :] == np.eye(n))
    for k in range(T-1):
        constr.append(Phi_x[n*(k+1):n*(k+1+1), :] == A@Phi_x[n*k:n*(k+1), :] + B@Phi_u[p*k:p*(k+1), :])
    constr.append(A@Phi_x[n*(T-1):, :] + B@Phi_u[p*(T-1):, :] == 0)

    # H2 constraint:
    # By Parseval's identity, this is equal (up to constants) to
    #
    # frobenius_norm(
    #   [ Q_sqrt*Phi_x[1] ;
    #     ...
    #     Q_sqrt*Phi_x[T] ;
    #     R_sqrt*Phi_u[1] ;
    #     ...
    #     R_sqrt*Phi_u[T]
    #   ]
    # ) <= htwo_cost
    # TODO: what is the best way to implement this in cvxpy?
    constr.append(
        cvx.norm(
            cvx.bmat(
                [[Q_sqrt@Phi_x[n*k:n*(k+1), :]] for k in range(T)] +
                [[R_sqrt@Phi_u[p*k:p*(k+1), :]] for k in range(T)]),
            'fro') <= htwo_cost)

    prob = cvx.Problem(cvx.Minimize(htwo_cost), constr)
    prob.solve(solver=cvx.SCS)

    if prob.status == cvx.OPTIMAL:
        #print("successfully solved!")
        Phi_x = np.array(Phi_x.value)
        Phi_u = np.array(Phi_u.value)
        return (True, prob.value, Phi_x, Phi_u, None, None)
    else:
        print("could not solve: {}".format(prob.status))
        return (False, None, None, None, None, None)
