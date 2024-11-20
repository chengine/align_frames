#%%
import torch
import pdb

def abs_orientation(X, Y):
    """
    Determine the optimal transformation that brings points from
    X's reference frame to points in Y's.
    T(x) = c * Rx + t where x is a point 3x1, c is the scaling, 
    R is a 3x3 rotation matrix, and t is a 3x1 translation.

    This is based off of "Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns" Umeyama 1991.

    Inputs:
        X - Tensor with dimension N x m
        Y - Tensor with dimension N x m
    Outputs:
        c - Scalar scaling constant
        R - Tensor 3x3 rotation matrix
        t - Tensor 3
    """

    N, m = X.shape

    mux = torch.mean(X, 0, True)
    muy = torch.mean(Y, 0, True)
    
    Yd = (Y - muy).unsqueeze(-1)
    Xd = (X - mux).unsqueeze(1)
    sx = torch.sum(torch.norm(Xd.squeeze(), dim=1) ** 2) / N
    Sxy = (1 / N) * torch.sum(torch.matmul(Yd, Xd), dim=0)

    if torch.linalg.matrix_rank(Sxy) < m:
        raise NameError("Absolute orientation transformation does not exist!")

    U, D, Vt = torch.linalg.svd(Sxy, full_matrices=True)
    S = torch.eye(m).to(dtype=Vt.dtype)
    if torch.linalg.det(Sxy) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    c = torch.trace(torch.diag(D) @ S) / sx
    t = muy.T - c * (R @ mux.T)

    return c, R, t.squeeze()

X = torch.randn(50, 3)

test_t = torch.randn(1, 3)
test_R, _ = torch.linalg.qr(torch.randn(3,3))
test_c = torch.rand(1)

Y = test_c * torch.matmul(test_R, X.unsqueeze(-1)).squeeze() + test_t

res_c, res_R, res_t = abs_orientation(X, Y)
