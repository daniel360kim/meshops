import torch
from typing import Union

"""
Hashmap of which direction to steal from and which direction to give to.
"""
RANGES = {
    (0, 90): ((0, 1), (-1, 0)),
    (90, 180): ((-1, 0), (0, -1)),
    (180, 270): ((0, -1), (1, 0)),
    (270, 360): ((1, 0), (0, 1)),
}

AXES = {
    0: (0, 1),
    90: (-1, 0),
    180: (0, -1),
    270: (1, 0)
}


def apply_padding(tensor: torch.Tensor, pad_type: Union[str, float]) -> torch.Tensor:
    if pad_type == 'copy':
        
        padded = torch.nn.functional.pad(tensor, (1, 1, 1, 1), mode='constant', value=0)
        for i in range(padded.shape[0]):
            padded[i][0] = padded[i][1]
            padded[i][-1] = padded[i][-2]
            
        for j in range(padded.shape[1]):
            padded[0][j] = padded[1][j]
            padded[-1][j] = padded[-2][j]
            
        return padded
        

    if type(pad_type) == float and 0 <= pad_type <= 1:
        # constant value
        return torch.nn.functional.pad(tensor, (1, 1, 1, 1), mode='constant', value=pad_type)

    else:
        raise ValueError("Invalid pad_type. It should be either 'copy' or a float value between 0 and 1.")



def apply_velocity(tensor: torch.Tensor, degrees: float, device, padding: Union[str, float] = 'copy') -> torch.Tensor:
    """
    If `padding` == 'copy', then the padding on the outside of the tensor will be filled with the values of the border.
    Another option is to set `padding` to a float value, which will fill the padding completely.
    """

    tensor = tensor.to(device)

    abs_component_horiz = torch.abs(torch.cos(torch.deg2rad(degrees))) # changes along a row
    abs_component_vert = torch.abs(torch.sin(torch.deg2rad(degrees))) # changes along a column

    ratio_horiz = abs_component_horiz / (abs_component_horiz + abs_component_vert)
    ratio_vert = abs_component_vert / (abs_component_horiz + abs_component_vert)
    
    steal_direction: ... # info about the cells to check when pulling heat from the direction of velocity
    get_robbed_direction: ... # info about the cells to check when giving heat to the direction of velocity
    
    flag = False
    for _range in RANGES:
        if degrees > _range[0] and degrees < _range[1]: # strictly between (axes are different)
            steal_direction, get_robbed_direction = RANGES[_range]
            flag = True
            break
    if not flag:
        for _angle in AXES:
            if degrees == _angle:
                steal_direction, get_robbed_direction = AXES[_angle]
                break
            
    if padding == 'copy':
        tensor = apply_padding(tensor, 'copy')
    elif type(padding) == float and 0 <= padding and 1 >= padding:
        tensor = apply_padding(tensor, padding)
    else:
        raise ValueError("Padding must be either 'copy' or a float value between 0 and 1.")

    # generating addition tensor, subtract from this after completion
    adjusted = tensor.clone()
    
    # Example corner:
    #
    #   Z       Y    Y2     
    #       _____________
    #       |>CURR|
    #   X   |  A  |  B
    #       |------------
    #   X2  |  C  |  D
    #       |     |

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            
            _steal_x = ratio_horiz * torch.abs(
                tensor[i+steal_direction[0][0], j+steal_direction[0][1]] - # coordinates of neighbor 1
                tensor[i,j] # coordinates of A
            )

            _steal_y = ratio_vert * torch.abs(
                tensor[i+steal_direction[1][0], j+steal_direction[1][1]] - # coordinates of neighbor 2
                tensor[i,j] #coordinates of A
            )

            adjusted[i, j] = tensor[i, j] + _steal_x + _steal_y

    # subtract
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            _delta_b = ratio_horiz * torch.abs(
                tensor[i+get_robbed_direction[0][0], j+get_robbed_direction[0][1]] - # coordinates of neighbor 1
                tensor[i,j] # coordinates of A
            )

            _delta_c = ratio_vert * torch.abs(
                tensor[i+get_robbed_direction[1][0], j+get_robbed_direction[1][1]] - # coordinates of neighbor 2
                tensor[i,j] #coordinates of A
            )

            adjusted[i, j] = tensor[i, j] - _delta_b - _delta_c

    return adjusted


