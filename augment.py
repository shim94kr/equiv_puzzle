import torch
from typing import Tuple

def kabsch_torch_batched(P, Q):
    """
    Source: https://hunterheidenreich.com/posts/kabsch_algorithm/
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute the covariance matrix
    H = torch.matmul(P.transpose(1, 2), Q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(P, R.transpose(1, 2)) - Q), dim=(1, 2)) / P.shape[1])

    return R, rmsd


class AugmentPipe:
    def __init__(self, 
                 shape_canonicalization = True):
        super().__init__()
        self.shape_canonicalization = shape_canonicalization

    def __call__(self, data):
        # shape canonicalization
        if self.shape_canonicalization:
            vertice = data['vertices']
            
            # pca
            U, S, V = torch.pca_lowrank(vertice, q=2)
            P1, P2 = [v.squeeze(-1) for v in V.chunk(chunks=2, dim=-1)]
            P3 = torch.cross(P1, P2, dim=1)
            # get rotaion matrix
            R, _ = kabsch_torch_batched(P1, P3)
            # applying rotation matrix
            data['vertices'] = torch.matmul(R, vertice)

        return data
