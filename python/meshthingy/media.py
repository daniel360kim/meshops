import torch
import stringcolor
from time import sleep
import colorsys
import math
from log import Log
from log import IterativeFile
from GIFmake import draw_gif
from timing import Timer
import uuid
import numpy as np
from PIL import Image
from debug_progress_bar import _get_progress_string
import matplotlib.colors as colors

if torch.cuda.is_available():
    print("CUDA is available, using GPU")
    device = torch.device("cuda")
else:
    print("CUDA is not available, using CPU")
    device = torch.device("cpu")

class ConductiveBar:
    def __init__(self, length: int = 50):
        self.mesh = torch.Tensor(length).fill_(0)
    
    def add_zone(self, start: int, end: int, temp: float = 0.5):
        for i in range(start, end):
            self.mesh[i] = temp 
            
    def get_iterable(self):
        return self.mesh
    
class ConductiveSurface:
    def __init__(self, length: int, width: int, default_temp: float = 0):
        self.mesh = torch.Tensor(length, width).fill_(default_temp)
    
    def heat_square(self, loc: tuple = (0, 0), radius: int = 2, temperature: float = 0.5) -> None:
        """
        Heats a "square" of mesh around `loc` to `temperature`.
        """
        padded = torch.nn.functional.pad(self.mesh, (radius, radius, radius, radius), mode='constant', value=0)
        
        # change the square region around loc to temperature
        row_region_start = loc[0] - radius
        row_region_end = loc[0] + radius + 1
        col_region_start = loc[1] - radius
        col_region_end = loc[1] + radius + 1
        
        padded[row_region_start:row_region_end,col_region_start:col_region_end] = temperature
        
        # unpad
        self.mesh = padded[radius:-radius, radius:-radius]
            
    def get_iterable(self) -> torch.Tensor:
        return self.mesh
    
    def _import_from_image(self, image_path: str) -> None:
        """
        Imports a 2D image from `image_path` and sets the mesh to the grayscale values of the image.
        """
        img = Image.open(image_path)
        img = img.resize((self.mesh.shape[0], self.mesh.shape[1]))
        
        #convert all to grayscale
        arr = np.mean(np.array(img), axis=2) / 255
        
        self.mesh = torch.Tensor(arr)

# 3d framework
class NDSquareMesh:
    def __init__(self, dimensions: tuple, default_temp: float = 0):
        self.mesh = torch.Tensor(dimensions).fill_(default_temp).to(device)
        self.dimensions = dimensions
        self.num_dimensions = len(dimensions)
        self.history = []
        
        self._pad_dims = lambda mask_radius: tuple([mask_radius] * 2 * self.num_dimensions)
    
    def heat_region(self, center: tuple, radius: int = 2, temperature: float = 0.5) -> None:
        """
        Heats a "N-dimensional cube" of mesh around `center` to `temperature`.
        """
        
        # padded = torch.nn.functional.pad(self.mesh, self._pad_dims(1), mode='constant', value=0)
        
        # change the square region around loc to temperature
        _nd_slice_obj = []
        
        for i in range(self.num_dimensions):
            _nd_slice_obj.append(slice(center[i] - radius, center[i] + radius + 1))

        self.mesh[*_nd_slice_obj] = temperature        
         
        # unpad
        _center_slice = [slice(radius,-radius)]*self.num_dimensions
        self.mesh = self.mesh[*_center_slice]
            
    def get_iterable(self) -> torch.Tensor:
        return self.mesh

    def run_timestep(self, conductivity_factor: float = 1) -> None:
        # save current state before moving on
        self.history.append(self.mesh.clone())
        
        # Move tensors to GPU
        
        # Create tensors for averaging with shifted versions of the heatmap tensor
        shifts = []
        for i in range(self.num_dimensions):
            shifts.append(torch.roll(self.mesh, shifts=1, dims=i))
            shifts.append(torch.roll(self.mesh, shifts=-1, dims=i))
        
        # Set certain edges to zero so numbers don't wrap around
        # TODO - context-aware padding,
        # XXX OR JUST HANDLE THE EDGES WITH A SEPARATE CASE IN THE AVERAGING FUNCTION
        #shifted_left[:, -1] = 0
        #shifted_right[:, 0] = 0
        #shifted_up[-1, :] = 0
        #shifted_down[0, :] = 0
        
        # Apply padding for the edge cases
        # TODO - context-aware padding
        padded_shifts = []
        _pad_dims = self._pad_dims(self.num_dimensions)
        padded_mesh = torch.nn.functional.pad(self.mesh, _pad_dims, mode='constant', value=0)
        
        for tens in shifts:
            # probably have a custom pad function that handles edges better, or just handle the edges in the averaging function
            padded_shifts.append(torch.nn.functional.pad(tens, _pad_dims, mode='constant', value=0))

        # Calculate the average using tensors and avoid iteration
        # Average the 3x3 square around each element
        # TODO - add support for weighted averages
        avg = (sum(padded_shifts)+padded_mesh) / (2 * self.num_dimensions + 1)
        
        # Remove padding
        _remove_pad_slices = [slice(1,-1)]*self.num_dimensions
        avg = avg[*_remove_pad_slices]
        
        # Move the result back to CPU if needed
        avg = avg.cpu()
        
        return avg
    
    def _get_mask(conductivity_factor: float = 1):
        """
        Returns a mask for the given conductivity factor.
        """
        return torch.Tensor(
            [
                [conductivity_factor/3, conductivity_factor, conductivity_factor/3],
                [conductivity_factor  , 1                  , conductivity_factor],
                [conductivity_factor/3, conductivity_factor, conductivity_factor/3]
            ]
        ).to(device)
        
    def _import_from_image(self, image_path: str) -> None:
        """
        Imports a 2D image from `image_path` and sets the mesh to the grayscale values of the image.
        """
        img = Image.open(image_path)
        img = img.resize((self.mesh.shape[0], self.mesh.shape[1]))
        
        #convert all to grayscale
        arr = np.mean(np.array(img), axis=2) / 255
        
        self.mesh = torch.Tensor(arr)