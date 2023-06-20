import torch
from typing import Iterable, SupportsIndex, Union

def _index_out_of_bounds(arr: Union[Iterable, SupportsIndex], row: int, col: int) -> bool:
    """
    Checks if a row and column are out of bounds of a 2D array.
    """
    return row < 0 or col < 0 or row >= len(arr) or col >= len(arr[row])


def weighted_pool(arr: Union[torch.Tensor, SupportsIndex], row: int, col: int, kernel: torch.Tensor) -> list:   
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    kernel = kernel.to(device)
    arr = arr.to(device)
    
    with torch.cuda.device(device):
        weighted_sum = 0.0
        kernel_weights_used = 0.0
        
        _kernel_lrow = kernel.shape[0] // 2 # radius along rows (horizontal), also the center row of the kernel
        _kernel_lcol = kernel.shape[1] // 2 # radius along cols (vertical), also the center col of the kernel
        
        #arr_interior = arr[_kernel_lrow:-_kernel_lrow, _kernel_lcol:-_kernel_lcol]
        
        # optimize to only weighted pool the edges, not going through all the interior
        for mov_row in range(-_kernel_lrow, _kernel_lrow + 1):
            for mov_col in range(-_kernel_lcol, _kernel_lcol + 1):
                if _index_out_of_bounds(arr, row + mov_row, col + mov_col):
                    continue
                
                weight = kernel[mov_row + _kernel_lrow][mov_col + _kernel_lcol]
                _dot = weight * arr[row + mov_row][col + mov_col]
                weighted_sum += _dot
                kernel_weights_used += weight
            
    return weighted_sum / kernel_weights_used

"""
def weighted_pool(
    arr: torch.Tensor | SupportsIndex, 
    row: int, 
    col: int, 
    kernel: torch.Tensor, 
    device: torch.device = torch.device('cpu')
    ) -> float:
    
    _kernel_lrow = kernel.shape[0] // 2
    _kernel_lcol = kernel.shape[1] // 2
    
    interior = (arr[_kernel_lrow:-_kernel_lrow, _kernel_lcol:-_kernel_lcol] * 100).long()
    _scaled_kernel = (kernel * 100).long()
    
    
    torch.nn.functional.conv2d(
        input=interior.unsqueeze(0).unsqueeze(0).to(device),
        weight=_scaled_kernel.unsqueeze(0).unsqueeze(0).to(device),
    )

    rows = torch.arange(row - _kernel_lrow, row + _kernel_lrow + 1).unsqueeze(1)
    cols = torch.arange(col - _kernel_lcol, col + _kernel_lcol + 1).unsqueeze(0)

    pooled_array = arr[rows, cols]
    print(f"Size of pooled array: {pooled_array.shape}, size of kernel: {kernel.shape}")
    weighted_array = pooled_array * kernel

    weighted_sum = weighted_array.sum()

    kernel_weights_used = kernel.sum()

    result = weighted_sum / kernel_weights_used

    return result.item()
"""

"""
def weighted_pool(arr: torch.Tensor | SupportsIndex, row: int, col: int, kernel: torch.Tensor) -> float:
    
    Finds the weighted average of a specified cell's neighbors, based on `kernel`.
    Averages are calculated by (neighborhood of cell * kernel) / sum(kernel)
    + If kernel is partially out of bounds, a partial average is calculated; no padding will
    be added (neighborhood of cell that are in bounds * kernel in bounds) / sum(kernel in bounds)
    
    _kernel_lrow = kernel.shape[0] // 2  # radius along rows (horizontal), also the center row of the kernel
    _kernel_lcol = kernel.shape[1] // 2  # radius along cols (vertical), also the center col of the kernel
    
    # Get the indices of the neighborhood cells
    indices_row = torch.arange(row - _kernel_lrow, row + _kernel_lrow + 1, device=arr.device)
    indices_col = torch.arange(col - _kernel_lcol, col + _kernel_lcol + 1, device=arr.device)
    
    # Ensure the indices are within bounds
    indices_row = torch.clamp(indices_row, 0, arr.shape[0] - 1)
    indices_col = torch.clamp(indices_col, 0, arr.shape[1] - 1)
    
    # Extract the neighborhood using indexing
    neighborhood = arr[indices_row[:, None], indices_col]
    
    # Apply the kernel and calculate the weighted sum
    weighted_sum = torch.sum(neighborhood * kernel)
    
    # Calculate the sum of kernel weights used
    kernel_weights_used = torch.sum(kernel)
    
    return weighted_sum / kernel_weights_used
"""