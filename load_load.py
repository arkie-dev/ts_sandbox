import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def load_load(a_ptr, y, b_ptr, size0, stride0, num0, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < size0

    idx = tl.load(y + offsets, mask=mask)

    x = tl.load(a_ptr + idx, mask=mask, other=0.0)

    result= x *2.0
    tl.store(b_ptr + offsets, result, mask=mask)



x = torch.rand(1024)
y = torch.rand(1024).int()
y = torch.randint(0, 1023, (1024,), dtype=torch.int32)
output = torch.empty(1024)
n_elements = output.numel()
load_load[(1,)](x, y, output, n_elements, 16, 8, BLOCK_SIZE=1024)

print(output)