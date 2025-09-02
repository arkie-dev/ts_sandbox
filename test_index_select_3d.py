import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def index_select_3d_kernel(
    input_ptr,
    output_ptr,
    indices,
    stride_i,
    stride_m,
    stride_n,
    stride_k,
    o_stride_m,
    o_stride_n,
    o_stride_k,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_offsets = tl.arange(0, BLOCK_I)
    n_indices = tl.load(indices + n_offsets)

    m_offsets = tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)

    input_offsets = (
        m_offsets[:, None, None] * stride_m
        + n_indices[None, :, None] * stride_n
        + k_offsets[None, None, :] * stride_k
    )

    input_pointers_0 = input_ptr + input_offsets
    data = tl.load(input_pointers_0)

    out_offsets = (
        m_offsets[:, None, None] * o_stride_m
        + n_offsets[None, :, None] * o_stride_n
        + k_offsets[None, None, :] * o_stride_k
    )
    tl.store(output_ptr + out_offsets, data)


def index_select_3d(input_tensor, indices, dim):
    M, N, K = input_tensor.shape
    R = indices.shape[0]
    output_tensor = torch.full(
        (M, R, K), -1, dtype=input_tensor.dtype, device=input_tensor.device
    )
    stride_i = indices.stride(0)
    stride_m = input_tensor.stride(0)
    stride_n = input_tensor.stride(1)
    stride_k = input_tensor.stride(2)
    o_stride_m = output_tensor.stride(0)
    o_stride_n = output_tensor.stride(1)
    o_stride_k = output_tensor.stride(2)
    index_select_3d_kernel[1,](
        input_tensor,
        output_tensor,
        indices,
        stride_i,
        stride_m,
        stride_n,
        stride_k,
        o_stride_m,
        o_stride_n,
        o_stride_k,
        BLOCK_I=R,
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    return output_tensor


def test_index_select_3d(device):
    M, N, K = 4, 4, 4
    input_tensor = torch.randn(M, N, K, device=device)
    indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    dim = 1  # Dimension to index along
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    output_ref = torch.index_select(input_tensor, dim, indices)
    output_triton = index_select_3d(input_tensor, indices, dim)
    torch.testing.assert_close(output_triton, output_ref)

test_index_select_3d('cpu')