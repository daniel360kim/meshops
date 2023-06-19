import torch
import stringcolor
from time import sleep
import colorsys
import math
from iterate import MetalBar
from datetime import datetime
import os

if torch.cuda.is_available():
    print("CUDA is available, using GPU")
else:
    print("CUDA is not available, using CPU")

ARRAY_SIZE = 90

## Create new csv file 
file_iterator = 0
while(os.path.exists(f"tests/export_data_{ARRAY_SIZE}_{file_iterator}.csv")):
    file_iterator += 1
csv_file = open(f"tests/export_data_{ARRAY_SIZE}_{file_iterator}.csv", "w")

## Log file
with open("tests/log.txt", "w") as log_file:
    log_file.write("Log file for csv file: ")
    log_file.write("export_data_{ARRAY_SIZE}_{file_iterator}.csv\n")
    log_file.write(csv_file.name)
    log_file.write(f"Test run at {datetime.now()}\n")
    log_file.write("Hardware:\n")
    if torch.cuda.is_available():
        log_file.write(f"    GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        log_file.write("    GPU: None\n")
    log_file.write(f"    Array size: {ARRAY_SIZE}\n")

    for i in range(3):
        log_file.write(".\n")


# convert to 1d tensor
heatmap_random = MetalBar(ARRAY_SIZE)
heatmap_random.add_zone(20, 40, 1)

heatmap_random = heatmap_random.get_iterable()

def runtimestep(heatmap: torch.Tensor, time_counter):
    # Create tensors for averaging with shifted versions of the heatmap tensor
    shifted_left = torch.roll(heatmap, shifts=1)
    shifted_right = torch.roll(heatmap, shifts=-1)
    
    # Apply padding for the edge cases
    padded_heatmap = torch.nn.functional.pad(heatmap, (1, 1), mode='constant', value=0)
    padded_shifted_left = torch.nn.functional.pad(shifted_left, (1, 1), mode='constant', value=0)
    padded_shifted_right = torch.nn.functional.pad(shifted_right, (1, 1), mode='constant', value=0)
    
    # Calculate the average using tensors and avoid iteration
    avg = (padded_shifted_left + padded_heatmap + padded_shifted_right) / 3.0
    
    _aux_tensor = avg[1:-1]  # Remove the padding from the result
    
    print_str_representation(_aux_tensor, time_counter)
    # f.write(f"{','.join([format(val, '.3f') for val in _aux_tensor.tolist()])}\n")

    return _aux_tensor


def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def print_str_representation(arr: torch.Tensor, time_counter) -> None:
    if torch.cuda.is_available():
        arr = arr.to(torch.device("cpu"))  # Move tensor to CPU
    else:
        arr = arr.to(torch.device("cpu"))  # Move tensor to CPU
    colored_chars = [get_color_escape(*val_to_rgb(val), background=True) + " " for val in arr.cpu().tolist()]
    totalstr = ''.join(colored_chars)
    print(f"{totalstr}\033[0m Time: {time_counter}; Hottest: {round(arr.max().item(), 3)}; Coolest: {round(arr.min().item(), 3)}", end="\r")


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
    #print(f"\nIteration: {time_counter}; Shape={heatmap_random.shape}", end="\r")

