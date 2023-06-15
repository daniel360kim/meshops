import torch
import stringcolor
from time import sleep
import colorsys
import math

class MetalBar:
    def __init__(self, length: int = 50):
        self.atoms = torch.Tensor(length).fill_(0)
    
    def add_zone(self, start: int, end: int, temp: float = 0.5):
        for i in range(start, end):
            self.atoms[i] = temp 
            
    def get_iterable(self):
        return self.atoms

# convert to 1d tensor
heatmap_random = MetalBar(10000)

# add zones
heatmap_random.add_zone(30, 40, 1)

heatmap_random = heatmap_random.get_iterable()

def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def runtimestep(heatmap: torch.Tensor, time_counter):
    _aux_tensor = torch.zeros(len(heatmap))
    for mid in range(len(heatmap)):
        # edges:
        if mid == 0:
            avg = (heatmap[0] + heatmap[1]) / 2
        elif mid == len(heatmap) - 1:
            avg = (heatmap[-1] + heatmap[-2]) / 2
        else:
            avg = (heatmap[mid - 1] + heatmap[mid] + heatmap[mid + 1]) / 3
        _aux_tensor[mid] = avg        
    
    print_str_representation(_aux_tensor, time_counter)
    #f.write(f"{','.join([format(val, '.3f') for val in _aux_tensor.tolist()])}\n")
    
    return _aux_tensor
        
def print_str_representation(arr: torch.Tensor, time_counter) -> None:
    # colormap: blue = 0, red = 1
    totalstr = ""
    for val in arr:
        colored_char = get_color_escape(*val_to_rgb(val), background=True) + " "
        totalstr += colored_char
    print(str(totalstr) + f"\033[0m Time: {time_counter}; Hottest: {round(max(arr).item(), 3)}; Coolest: {round(min(arr).item(), 3)}", end="\r")

def val_to_rgb(val: float) -> tuple[int]:
    """
    maps 0-1 -> rgb255 linearly
    """
    # desmos eq: \frac{\ln\left(\sqrt{x}+1\right)}{1.1}
    # scaled_val = math.log(math.sqrt(val) + 1) / 1.1
    scaled_val = -0.693*val+0.693
    
    _unscaled = colorsys.hsv_to_rgb(scaled_val, 1, 1)
    return tuple([int(val * 255) for val in _unscaled])


time_counter = 0
#print()
#print_str_representation(heatmap_random, time_counter)
#with open("export_data.csv", "w") as f:
while True:
    time_counter += 1
    heatmap_random = runtimestep(heatmap_random, time_counter)
    print(f"Iteration: {time_counter}", end="\r")

