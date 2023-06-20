import torch
import stringcolor
from time import sleep
import time
import colorsys
import math
from iterate import MetalBar
from datetime import datetime
import os

if torch.cuda.is_available():
    print("CUDA is available, using GPU")
else:
    print("CUDA is not available, using CPU")

ARRAY_SIZE = 1000

## Create new csv file 
file_iterator = 0
while(os.path.exists(f"tests/export_data_{ARRAY_SIZE}_{file_iterator}.csv")):
    file_iterator += 1
csv_file = open(f"iteration.csv", "w")

# convert to 1d tensor
heatmap_random = MetalBar(ARRAY_SIZE)
heatmap_random.add_zone(20, 40, 1)

heatmap_random = heatmap_random.get_iterable()

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    progress = 0
    if total != 0:
        progress = iteration / float(total)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {progress:.1%} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def runtimestep(heatmap: torch.Tensor, time_counter):
    size = heatmap.size()[0]
    avg = torch.zeros_like(heatmap)
    
    for i in range(size):
        if i == 0:
            avg[i] = (heatmap[i] + heatmap[i+1]) / 2.0
        elif i == size - 1:
            avg[i] = (heatmap[i-1] + heatmap[i]) / 2.0
        else:
            avg[i] = (heatmap[i-1] + heatmap[i] + heatmap[i+1]) / 3.0
    
    _aux_tensor = avg
    
    print_str_representation(_aux_tensor, time_counter)
    # f.write(f"{','.join([format(val, '.3f') for val in _aux_tensor.tolist()])}\n")
    
    return _aux_tensor

def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def print_str_representation(arr: torch.Tensor, time_counter) -> None:
    colored_chars = []
    for val in arr:
        rgb = val_to_rgb(val)
        color_escape = get_color_escape(*rgb, background=True)
        colored_chars.append(color_escape + " ")
    
    totalstr = ''.join(colored_chars)
    #print(f"{totalstr}\033[0m Time: {time_counter}; Hottest: {round(arr.max().item(), 3)}; Coolest: {round(arr.min().item(), 3)}", end="\r")

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
for i in range(1000):
    time_counter += 1
    t1 = time.perf_counter_ns()
    heatmap_random = runtimestep(heatmap_random, time_counter)
    t2 = time.perf_counter_ns() - t1

    progress_bar(i, 1000, prefix='Progress:', suffix='Complete', length=50)

    #write time to csv
    csv_file.write(f"{t2}\n")


    #print(f"\nIteration: {time_counter}; Shape={heatmap_random.shape}", end="\r")

