import torch
import galois
import numpy as np

def gf2_rank(mat: torch.Tensor) -> int:
    """Rank over GF(2)."""
    arr = mat.detach().to("cpu").numpy().astype(np.int8, copy=False)
    if arr.size == 0:
        return 0
    GF2 = galois.GF(2)
    return GF2(arr).row_space().shape[0]


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    A = galois.GF(2)(matrix)
    return np.asarray(np.linalg.inv(A), dtype=np.int8)


def _quotient_basis(GF2, null_space, row_space):
    current = GF2(row_space).row_space()
    logicals = []

    for v in GF2(null_space):
        candidate = GF2(np.vstack([current, v]))
        basis = candidate.row_space()

        if basis.shape[0] > current.shape[0]:
            logicals.append(v)
            current = basis

    return (
        GF2(np.vstack(logicals))
        if logicals
        else GF2.Zeros((0, null_space.shape[1]))
    )


def _pair_logical_bases(logical_x, logical_z, n_data, k, device):
    """Transform logical X operators so that X @ Z.T = I over GF(2)."""

    # Ensure that the logical operators are 2D matrices:
    # (number of logical operators, number of physical qubits).
    logical_x = logical_x.reshape(-1, n_data)
    logical_z = logical_z.reshape(-1, n_data)

    # Compute the original pairing matrix over GF(2).
    M = ((logical_x.float() @ logical_z.T.float()) % 2).int()
    M_inv = torch.as_tensor(gf2_inverse(M.detach().cpu().numpy()), dtype=torch.long, device=device,)
    logical_x = (M_inv.float() @ logical_x.float()) % 2

    # Verify the result.
    assert torch.equal(
        (logical_x.float() @ logical_z.T.float()) % 2,
        torch.eye(k, dtype=torch.float32,device=device)
    ),"Logical X/Z pairing is not the identity matrix"

    return logical_x, logical_x.T, logical_z, logical_z.T


def logical_support_dimension(error_support: torch.Tensor, commuting_checks: torch.Tensor,
                               stabilizers: torch.Tensor, stabilizer_rank: int) -> int:
    # Indices where an error is present / absent.
    support = torch.where(error_support)[0]
    outside = torch.where(~error_support)[0]

    if support.numel() == 0:
        return 0

    normalizer_support_dim = support.numel() - gf2_rank(commuting_checks[:, support])
    stabilizer_support_dim = stabilizer_rank - gf2_rank(stabilizers[:, outside])

    return max(0, normalizer_support_dim - stabilizer_support_dim)


def get_logical_operators(H_x, H_z, n_data, k, device):
    GF2 = galois.GF(2)

    Hx = GF2(H_x.detach().cpu().numpy())
    Hz = GF2(H_z.detach().cpu().numpy())

    logical_z = torch.as_tensor(_quotient_basis(GF2, Hx.null_space(), Hz.row_space()), dtype=torch.long, device=device)
    logical_x = torch.as_tensor(_quotient_basis(GF2, Hz.null_space(), Hx.row_space()), dtype=torch.long, device=device)

    return _pair_logical_bases(logical_x, logical_z, n_data, k, device)