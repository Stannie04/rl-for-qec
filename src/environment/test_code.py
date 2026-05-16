import pytest
import torch

from src.environment.code import QLDPCCode


def make_code(Hx, Hz, logical_x, logical_z, k):
    code = QLDPCCode.__new__(QLDPCCode)
    code.H_x = torch.tensor(Hx, dtype=torch.float32)
    code.H_z = torch.tensor(Hz, dtype=torch.float32)
    code.logical_x = torch.tensor(logical_x, dtype=torch.float32)
    code.logical_z = torch.tensor(logical_z, dtype=torch.float32)
    code.k = k
    code.n_data = code.H_x.shape[1]
    return code


def test_assert_valid_code_accepts_nontrivial_css_code():
    # n=3, Hx empty, Hz has rank 2 => k = 1
    # logical_x = 111 is in ker(Hz)
    # logical_z = 100 is not in row(Hz), and commutes with Hx (empty)
    code = make_code(
        Hx=[[] , [],],  # replaced below with explicit shape
        Hz=[[1, 1, 0], [0, 1, 1]],
        logical_x=[[1, 1, 1]],
        logical_z=[[1, 0, 0]],
        k=1,
    )
    code.H_x = torch.zeros((0, 3), dtype=torch.float32)
    assert code._assert_valid_code() is True


def test_assert_valid_code_rejects_commutation_violation():
    # Hx * Hz^T = 1 mod 2
    code = make_code(
        Hx=[[1, 1, 0]],
        Hz=[[1, 0, 1]],
        logical_x=[[1, 0, 0]],
        logical_z=[[0, 1, 0]],
        k=1,
    )
    with pytest.raises(ValueError, match="do not commute"):
        code._assert_valid_code()


def test_assert_valid_code_rejects_wrong_k():
    code = make_code(
        Hx=[[0,0,0], [0,0,0]],
        Hz=[[1, 1, 0], [0, 1, 1]],
        logical_x=[[1, 1, 1]],
        logical_z=[[1, 0, 0]],
        k=2,  # should be 1
    )
    code.H_x = torch.zeros((0, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="expected k=1"):
        code._assert_valid_code()


def test_assert_valid_code_rejects_logical_in_stabilizer_row_space():
    # Hx row = 1100, Hz row = 0011, so k = 2
    # First logical_x is exactly in row(Hx), so it should fail independence check.
    code = make_code(
        Hx=[[1, 1, 0, 0]],
        Hz=[[0, 0, 1, 1]],
        logical_x=[[1, 1, 0, 0], [1, 0, 0, 0]],
        logical_z=[[0, 0, 1, 0], [0, 0, 0, 1]],
        k=2,
    )
    with pytest.raises(ValueError, match="not linearly independent"):
        code._assert_valid_code()