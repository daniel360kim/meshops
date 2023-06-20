import torch
import numpy as np
from PIL import Image
import itertools
from typing import Union

from log import Log
from log import IterativeFile
from timing import Timer
from errors import UnsupportedDimensionError
from weighted_pool import weighted_pool

def _index_out_of_bounds(arr, row, col):
    """
    Checks if a row and column are out of bounds of a 2D array.
    """
    return row < 0 or col < 0 or row >= len(arr) or col >= len(arr[row])

def get_numbers_around_location(arr, row, col, radius=1):
    """
    Gets all numbers around a specified location in a 2D array as a list.
    If on edge, the list will contain less numbers.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        arr = arr.to(device)
    else:
        device = torch.device("cpu")

    nums = []

    with torch.cuda.device(device):
        for mov_row in range(-radius, radius + 1):
            for mov_col in range(-radius, radius + 1):
                if mov_row == 0 and mov_col == 0:
                    continue

                if _index_out_of_bounds(arr, row + mov_row, col + mov_col):
                    continue

                nums.append(arr[max(0, row + mov_row)][max(0, col + mov_col)])
    return nums

def _get_edge_indices(length, width, border_width):
    edge_indices = []
    for i in range(length):
        if i < border_width or i >= length - border_width:
            edge_indices.extend([(i, j) for j in range(width)])
        else:
            for j in range(border_width):
                edge_indices.extend([(i, j), (i, width - 1 - j)])
    return edge_indices

class ConductiveBar:
    def __init__(self, length=50):
        self.mesh = torch.Tensor(length).fill_(0)

    def add_zone(self, start, end, temp=0.5):
        for i in range(start, end):
            self.mesh[i] = temp

    def get_iterable(self):
        return self.mesh

class ConductiveSurface:
    def __init__(self, shape: tuple[int], default_temp: float = 0):
        length, width = shape
        self.mesh = torch.Tensor(length, width).fill_(default_temp)

    def heat_square(self, loc=(0, 0), radius=2, temperature=0.5):
        """
        Heats a "square" of mesh around `loc` to `temperature`.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        padded = torch.nn.functional.pad(self.mesh, (radius, radius, radius, radius), mode='constant', value=0)

        padded = padded.to(device)
        self.mesh = self.mesh.to(device)

        with torch.cuda.device(device):
            # change the square region around loc to temperature
            row_region_start = loc[0] - radius
            row_region_end = loc[0] + radius + 1
            col_region_start = loc[1] - radius
            col_region_end = loc[1] + radius + 1

            padded[row_region_start:row_region_end, col_region_start:col_region_end] = temperature

            # unpad
            self.mesh = padded[radius:-radius, radius:-radius]

    def get_iterable(self):
        return self.mesh

    def _import_from_image(self, image_path):
        """
        Imports a 2D image from `image_path` and sets the mesh to the grayscale values of the image.
        """
        img = Image.open(image_path)
        img = img.resize((self.mesh.shape[0], self.mesh.shape[1]))

        # convert all to grayscale
        arr = np.mean(np.array(img), axis=2) / 255

        self.mesh = torch.Tensor(arr)

# 3d framework
class NDSquareMesh:
    def __init__(self, *args):
        """
        # NDSquareMesh
        
        Represents an n-dimensional square mesh. Each cell can interact with its neighbors through
        user defined interactions.
        
        ## Overloads
        `NDSquareMesh(pytorch_tensor: torch.Tensor)`
        `NDSquareMesh(shape: tuple[int], default_temp: int)`
        """
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if len(args) == 1:
            if type(args[0]) == torch.Tensor:
                self.mesh = args[0].to(self.device)
            elif type(args[0]) == np.ndarray:
                self.mesh = torch.Tensor(args[0]).to(self.device)
        
        elif len(args) == 2 and type(args[0]) == tuple and type(args[1]) == int:
            # shape, default_temp
            shape, default_temp = args
            self.mesh = torch.full(shape, default_temp).to(self.device)
            
        else:
            raise TypeError("NDSquareMesh constructor accepts either `(tensor to convert): torch.Tensor` or `(shape): tuple[int], (default temp): int`")
            
        self.shape = self.mesh.shape
        self.dimensions = len(self.shape)
        #self.history = []
            
        self._pad_dims = lambda mask_radius: tuple([mask_radius] * 2 * self.dimensions)
    
    def set_region(self, center: tuple, radius: int = 2, temperature: float = 0.5) -> None:
        """
        Sets a "N-dimensional cube" of mesh around `center` to `temperature`.
        """
        
        # padded = torch.nn.functional.pad(self.mesh, self._pad_dims(1), mode='constant', value=0)
        
        # change the square region around loc to temperature
        _nd_slice_obj = []
        
        for i in range(self.dimensions):
            _nd_slice_obj.append(slice(center[0] - radius, center[0] + radius + 1))

        self.mesh[tuple(_nd_slice_obj)] = temperature
        
        # unpad
        # print(f"before unpad: {self.mesh.shape}")
        # _center_slice = [slice(radius,-radius)]*self.dimensions
        # self.mesh = self.mesh[*_center_slice]
        # print(f"not sliced slice dims: {self.mesh.shape}")

            
    def get_iterable(self):
        return self.mesh.to(self.device)

    def run_timestep(
        self, 
        kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]], 
        conductivity_factor: float = ...
        ) -> None:
        
        if self.dimensions != 2:
            raise NotImplementedError("Only 2D meshes are supported at the moment.")
        
        kernel_sum = sum([sum(row) for row in kernel])
        #interior_ranges = [
        #    slice(kernel.shape[i]//2, -kernel.shape[i]//2) for i in range(self.dimensions)
        #]
        _ker_tensor = torch.Tensor(kernel)
        

        with torch.cuda.device(self.device):
            # Only run this function for the exterior of the mesh
            # For the interior, use the dot product
            
            scaled = (100 * self.mesh).long().unsqueeze(0).unsqueeze(0).to(self.device)
            #print(f"scaled: \n{scaled}")
            scaled_weights = (_ker_tensor).long().unsqueeze(0).unsqueeze(0).to(self.device)
            
            new = torch.nn.functional.conv2d(
                scaled,
                scaled_weights,
            ) / (kernel_sum * 100)
            
            #new = new.float()/100
            
            new = torch.nn.functional.pad(new, (1, 1, 1, 1), mode='constant', value=0)
            new = new[0, 0]
            
            # get indicies of edges
            edge_indicies = _get_edge_indices(self.shape[0], self.shape[1], _ker_tensor.shape[0])
            
            for coords in edge_indicies:
                new[coords[0],coords[1]] = weighted_pool(
                    self.mesh,
                    coords[0],
                    coords[1],
                    _ker_tensor
                )
                
            self.mesh = new.to(self.device) #/ 100
    
    def _get_mask(self, conductivity_factor=1):
        """
        Returns a mask for the given conductivity factor.
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        with torch.cuda.device(device):
            mask = torch.tensor(
                [
                    [conductivity_factor/3, conductivity_factor, conductivity_factor/3],
                    [conductivity_factor  , 1                  , conductivity_factor],
                    [conductivity_factor/3, conductivity_factor, conductivity_factor/3]
                ]
            )
        return mask
        
    def _import_from_image(self, image_path):
        
        if self.dimensions != 2:
            raise UnsupportedDimensionError("Importing from an image is only supported for 2D meshes.")
        
        """
        Imports a 2D image from `image_path` and sets the mesh to the grayscale values (0-1) of the image.
        """
        img = Image.open(image_path)

        img = img.resize((self.mesh.shape[0], self.mesh.shape[1]))
        
        # convert all to grayscale
        arr = np.mean(np.array(img), axis=2) / 255
        
        self.mesh = torch.Tensor(arr).to(self.device)
