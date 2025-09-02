import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def load(in_ptr, t, out_ptr, size0, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < size0
    
    x = tl.load(in_ptr + offsets , mask=mask, other=0.0)

    result= x *2.0
    tl.store(out_ptr + offsets, result, mask=mask)



x = torch.rand(1024)
y = torch.randint(0, 32, (32,32), dtype=torch.int32)
print(y)
output = torch.empty(1024)
n_elements = output.numel()
load[(1,)](x, y, output, n_elements, BLOCK_SIZE=32)

print(output)