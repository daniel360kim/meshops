import torch
from media import NDSquareMesh

def apply_ndmask(tensor: torch.Tensor | NDSquareMesh, mask: torch.Tensor) -> torch.Tensor | NDSquareMesh:
    """
    Applies a mask to a tensor or NDSquareMesh. The mask must be the same shape as the tensor or mesh.
    """
    if type(tensor) == torch.Tensor:
        return tensor * mask
    elif type(tensor) == NDSquareMesh:
        tensor.mesh = tensor.get_iterable() * mask
        return tensor
    else:
        raise(TypeError("apply_ndmask accepts either `torch.Tensor` or `NDSquareMesh`"))