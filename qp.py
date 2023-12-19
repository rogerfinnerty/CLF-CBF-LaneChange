"""
Function for solving quadratic program 
"""

import cvxopt as cvx
import numpy as np

def qp_solver(h_mat, f_mat, a_cbf, b_cbf, solver='cvxopt'):
    """
    Quadratic program solver
    """
    h = cvx.matrix(h_mat)
    f = cvx.matrix(f_mat, tc='d')

    constraint_a = cvx.matrix(a_cbf)
    constraint_b = cvx.matrix(b_cbf)

    u = cvx.solvers.qp(h, f, constraint_a, constraint_b, solver=solver)

    return np.array(u['x'])
