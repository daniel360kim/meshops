import torch
import numpy as np
from PIL import Image
import itertools

from log import Log
from log import IterativeFile
from timing import Timer
from errors import UnsupportedDimensionError
from weighted_pool import weighted_pool

if torch.cuda.is_available():
    print("CUDA is available, using GPU")
    device = torch.device("cuda")
else:
    print("CUDA is not available, using CPU")
    device = torch.device("cpu")

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
    nums = []

    for mov_row in range(-radius, radius + 1):
        for mov_col in range(-radius, radius + 1):
            if mov_row == 0 and mov_col == 0:
                continue

            if _index_out_of_bounds(arr, row + mov_row, col + mov_col):
                continue

            nums.append(arr[max(0, row + mov_row)][max(0, col + mov_col)])
    return nums

class ConductiveBar:
    def __init__(self, length=50):
        self.mesh = torch.Tensor(length).fill_(0)

    def add_zone(self, start, end, temp=0.5):
        for i in range(start, end):
            self.mesh[i] = temp

    def get_iterable(self):
        return self.mesh

class ConductiveSurface:
    def __init__(self, shape, default_temp=0):
        length, width = shape
        self.mesh = torch.Tensor(length, width).fill_(default_temp)

    def heat_square(self, loc=(0, 0), radius=2, temperature=0.5):
        """
        Heats a "square" of mesh around `loc` to `temperature`.
        """
        padded = torch.nn.functional.pad(self.mesh, (radius, radius, radius, radius), mode='constant', value=0)

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
        user-defined interactions.
        
        ## Overloads
        `NDSquareMesh(pytorch_tensor: torch.Tensor)`
        `NDSquareMesh(shape: tuple[int], default_temp: int)`
        """
        
        if len(args) == 1:
            if type(args[0]) == torch.Tensor:
                self.mesh = args[0]
            elif type(args[0]) == np.ndarray:
                self.mesh = torch.Tensor(args[0])
        
        elif len(args) == 2 and type(args[0]) == tuple and type(args[1]) == int:
            # shape, default_temp
            shape, default_temp = args
            self.mesh = torch.full(shape, default_temp)
            
        else:
            raise TypeError("NDSquareMesh constructor accepts either `(tensor to convert): torch.Tensor` or `(shape): tuple[int], (default temp): int`")
            
        self.shape = self.mesh.shape
        self.dimensions = len(self.shape)
        #self.history = []
            
        self._pad_dims = lambda mask_radius: tuple([mask_radius] * 2 * self.dimensions)
    
    def set_region(self, center, radius=2, temperature=0.5):
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
        return self.mesh

    def run_timestep(self, kernel=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], conductivity_factor=...):
    if self.dimensions != 2:
        raise NotImplementedError("Only 2D meshes are supported at the moment.")

    _ker_tensor = torch.Tensor(kernel)
    # apply the kernel to each cell (& its neighbors) in the mesh
    for coords in itertools.product(*[range(i) for i in self.shape]):
        print(f"{coords[0] * 100 + coords[1]}", end="\r")
        self.mesh[tuple(coords)] = weighted_pool(self.mesh, coords[0], coords[1], _ker_tensor)

    # Create tensors for averaging with shifted versions of the heatmap tensor
    # shifts = []
    # for i in range(self.dimensions):
    #     shifts.append(torch.roll(self.mesh, shifts=1, dims=i))
    #     shifts.append(torch.roll(self.mesh, shifts=-1, dims=i))
    #
    # Set certain edges to zero so numbers don't wrap around
    # TODO - context-aware padding,
    # XXX OR JUST HANDLE THE EDGES WITH A SEPARATE CASE IN THE AVERAGING FUNCTION
    # shifted_left[:, -1] = 0
    # shifted_right[:, 0] = 0
    # shifted_up[-1, :] = 0
    # shifted_down[0, :] = 0
    #
    # Apply padding for the edge cases
    # TODO - context-aware padding
    # padded_shifts = []
    # _pad_dims = self._pad_dims(1)
    # padded_mesh = torch.nn.functional.pad(self.mesh, _pad_dims, mode='constant', value=0)
    #
    # for tens in shifts:
    #     # probably have a custom pad function that handles edges better, or just handle the edges in the averaging function
    #     padded_shifts.append(torch.nn.functional.pad(tens, _pad_dims, mode='constant', value=1))
    #
    # Calculate the average using tensors and avoid iteration
    # Average the 3x3 square around each element
    # TODO - add support for weighted averages
    # avg = (sum(padded_shifts)+padded_mesh) / (2 * self.dimensions + 1)
    #
    # Remove padding
    # _remove_pad_slices = [slice(1,-1)]*self.dimensions
    # avg = avg[*_remove_pad_slices]
    #
    # Move the result back to CPU if needed
    # avg = avg.cpu()
    #
    # self.mesh = avg

    
    def _get_mask(conductivity_factor=1):
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
        
        self.mesh = torch.Tensor(arr)
